import glob
from typing import Iterable, Union, Any
import torch, os, random
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
import numpy as np
from nnunet.training.data_augmentation import downsampling
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.training.feature_selector import FeatureSelector

from enum import Enum
import os
from os.path import join, exists
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Set, Tuple
from abc import ABC, abstractmethod
import pickle
from tqdm import tqdm
import shutil
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileTracker:
    def __init__(self, base_path: str):
        """
        Track and log file operations and issues.

        Args:
            base_path: Base path for log files
        """
        self.base_path = base_path
        self.missing_files: Set[str] = set()
        self.corrupted_files: Set[str] = set()
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up file logger
        self.log_file = os.path.join(
            self.log_dir,
            f"feature_rehearsal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)

    def log_missing_file(self, file_path: str):
        """Log missing file and add to tracking set."""
        self.missing_files.add(file_path)
        logger.error(f"Missing file: {file_path}")

    def log_corrupted_file(self, file_path: str, error: Exception):
        """Log corrupted file and add to tracking set."""
        self.corrupted_files.add(file_path)
        logger.error(f"Corrupted file {file_path}: {str(error)}")

    def save_report(self):
        """Save summary report of file issues."""
        report_path = os.path.join(self.log_dir, "file_issues_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== Missing Files ===\n")
            for file in sorted(self.missing_files):
                f.write(f"{file}\n")
            f.write("\n=== Corrupted Files ===\n")
            for file in sorted(self.corrupted_files):
                f.write(f"{file}\n")
        logger.info(f"File issues report saved to {report_path}")

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

class FeatureSelectionMethod(Enum):
    RANDOM = "random"
    CHINESE_WHISPERS = "chinese_whispers"
    HDBSCAN = 'HDBSCAN'

class DatasetBuffer:
    def __init__(self,
                 retention_ratio: float = 0.5,
                 batch_size: int = 32):
        self.retention_ratio = retention_ratio
        self.batch_size = batch_size
        self.valid_indices: List[int] = []
        self.processed_files: Set[str] = set()
        self.data_patches: List[str] = []  # Store the actual files for this dataset
        self.file_to_index: Dict[str, int] = {}  # Map files to their indices


class SparsityAnalyzer:
    def __init__(self):
        self.dataset_stats = {}

    def update_stats(self, feature_map, dataset_id):
        """Update running statistics"""
        if dataset_id not in self.dataset_stats:
            self.dataset_stats[dataset_id] = {
                'sum_negative': 0.0,
                'sum_zero': 0.0,
                'num_samples': 0
            }

        stats = self.dataset_stats[dataset_id]

        # Compute current ratios
        total_elements = feature_map.size
        curr_negative_ratio = (feature_map < 0).sum().item() / total_elements
        curr_zero_ratio = (feature_map == 0).sum().item() / total_elements

        # Accumulate sums
        stats['sum_negative'] += curr_negative_ratio
        stats['sum_zero'] += curr_zero_ratio
        stats['num_samples'] += 1

    def get_summary(self):
        """Get summary statistics for all datasets."""
        summary = {}
        for dataset_id, stats in self.dataset_stats.items():
            n = stats['num_samples']
            if n == 0:
                continue

            negative_ratio = stats['sum_negative'] / n
            zero_ratio = stats['sum_zero'] / n

            summary[dataset_id] = {
                'sparsity_ratio': negative_ratio + zero_ratio,
                'negative_ratio': negative_ratio,
                'zero_ratio': zero_ratio,
                'samples_analyzed': n
            }
        return summary

    def reset_dataset(self, dataset_id):
        """Reset statistics for a specific dataset."""
        if dataset_id in self.dataset_stats:
            del self.dataset_stats[dataset_id]

class FeatureRehearsalDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 deep_supervision_scales: list[list[float]],
                 target_type: FeatureRehearsalTargetType,
                 num_features: int,
                 retention_ratio: float = 0.05,
                 selection_method: FeatureSelectionMethod = FeatureSelectionMethod.RANDOM,
                 batch_size: int = 32,
                 cleanup_unused: bool = True,
                 new_task_idx: Optional[int] = None,
                 old_dict_from_file_name_to_task_idx: Optional[dict] = None,
                 load_skips: bool = True,
                 constant_skips: Optional[np.ndarray] = None,
                 selection_kwargs: Optional[Dict] = None,
                 random_state: int = 42) -> None:
        super().__init__()
        self.data_path = data_path
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features
        self.load_skips = load_skips
        self.constant_skips = constant_skips
        self.load_meta = False
        self.batch_size = batch_size
        self.cleanup_unused = cleanup_unused
        self.selection_method = selection_method
        self.retention_ratio = retention_ratio
        self.selection_kwargs = selection_kwargs or {}
        self.random_state = random_state
        self.sparsity_analyzer = SparsityAnalyzer()

        self.new_task_idx = new_task_idx
        self.old_dict_from_file_name_to_task_idx = old_dict_from_file_name_to_task_idx
        self.task_idx_array = []

        # Initialize file tracker
        self.file_tracker = FileTracker(data_path)

        # Initialize feature selector
        self._init_feature_selector()

        # Store separate buffers for each dataset
        self.dataset_buffers: Dict[int, DatasetBuffer] = {}
        self.current_dataset_id = 0

        # Global mapping of files to their dataset and index
        self.global_file_mapping: Dict[str, Tuple[int, int]] = {}

    def _verify_file_exists(self, file_path: str) -> bool:
        """Verify file exists and log if missing."""
        if not exists(file_path):
            self.file_tracker.log_missing_file(file_path)
            return False
        return True

    def _init_feature_selector(self):
        """Initialize feature selector based on selected method."""
        self.feature_selector = FeatureSelector(
                min_cluster_size=self.selection_kwargs.get('min_cluster_size', 5),
                random_state=self.random_state,
                distance_type=self.selection_kwargs.get('distance_type', 'manhattan'),
                batch_size=self.batch_size
            )


    def _get_all_related_files(self, patch_name):
        """Get all files related to a given patch."""
        files = []
        base_name = patch_name[:-4]  # Remove .npy extension

        # Feature files
        if not self.load_skips:
            for i in range(self.num_features):
                feature_file = join(self.data_path, "features", f"{base_name}_{i}.npy")
                if os.path.exists(feature_file):
                    files.append(feature_file)

        # Feature pickle files
        pkl_file = join(self.data_path, "feature_pkl", f"{base_name}.pkl")
        if os.path.exists(pkl_file):
            files.append(pkl_file)

        # GT files
        gt_file = join(self.data_path, "gt", patch_name)
        if os.path.exists(gt_file):
            files.append(gt_file)

        # Prediction files
        pred_file = join(self.data_path, "predictions", f"{base_name}_0.npy")
        if os.path.exists(pred_file):
            files.append(pred_file)

        return files

    def update_features(self, specific_files: Optional[List[str]] = None):
        """Update features and predictions with a new dataset."""

        try:
            data_patches = os.listdir(join(self.data_path, "gt"))
            data_patches.sort()
            if not self.load_skips:
                if self.constant_skips is None:
                    assert len(data_patches) > 0
                    # Load the pickle file first
                    loaded_data = load_pickle(join(self.data_path, "feature_pkl", data_patches[0][:-4] + ".pkl"))
                    features = loaded_data[:-1] # except bottleneck
                    self.constant_skips = []
                    for f in features:
                        zero_array = np.zeros_like(f)
                        self.constant_skips.append(zero_array)

            # Create new buffer for this dataset
            buffer = DatasetBuffer(
                retention_ratio=self.retention_ratio,
                batch_size=self.batch_size
            )

            # Get list of files to process
            if specific_files is None:
                current_files = os.listdir(join(self.data_path, "gt"))
                current_files.sort()
                files_to_process = [f for f in current_files
                                    if f not in self.global_file_mapping]
            else:
                files_to_process = [f for f in specific_files
                                    if f not in self.global_file_mapping]

            if not files_to_process:
                logger.info("No new features to process.")
                return

            # Load and process features and predictions
            valid_indices, valid_filenames = self._load_and_preprocess_features(files_to_process)

            # Update files_to_process to only include valid files
            valid_files = [files_to_process[i] for i in valid_indices]

            if int(len(valid_files) < 30):
                assert "Less than 30 features for this dataset!"

            # Calculate number of features to keep, at least 30
            num_features_to_keep = max(30, int(len(valid_files) * self.retention_ratio))

            # Create sample IDs for scores that match training records
            sample_ids = []
            for file_name in valid_filenames:
                base_name = file_name[:-4]
                parts = base_name.split('_')
                case_id = '_'.join(parts[:-3])
                z_coord = parts[-1]
                sample_id = f"{case_id}_{z_coord}"
                sample_ids.append(sample_id)

            selected_indices, metrics = self.feature_selector.select_samples(
                n_select=num_features_to_keep,
                method=self.selection_method.value,
                sample_ids=sample_ids,
            )

            logger.info(f"Selected {len(selected_indices)} features")

            # Update buffer with selected files
            buffer.valid_indices = selected_indices
            buffer.data_patches = valid_files

            # Update file mappings
            for idx, file_name in enumerate(valid_files):
                buffer.file_to_index[file_name] = idx

            # Store the buffer
            self.dataset_buffers[self.current_dataset_id] = buffer

            # Update global file mapping for selected files
            for idx in selected_indices:
                file_name = valid_files[idx]
                self.global_file_mapping[file_name] = (self.current_dataset_id, idx)

            logger.info(
                f"Dataset {self.current_dataset_id}: Processed {len(valid_files)} files, "
                f"kept {len(selected_indices)} features"
            )

            # Clean up unused files only from the current dataset
            if self.cleanup_unused:
                self._cleanup_unused_files(valid_files, selected_indices)

            # Save current file issues report
            self.file_tracker.save_report()

            # Increment dataset counter
            self.current_dataset_id += 1

        except Exception as e:
            logger.error(f"Error updating features: {str(e)}")
            raise

    def _cleanup_unused_files(self, all_files: List[str], valid_indices: List[int]):
        """Clean up files that weren't selected during deduplication for the current dataset only."""
        valid_files = {all_files[i] for i in valid_indices}
        files_to_backup = set(all_files) - valid_files

        if not files_to_backup:
            return

        backup_dir = join(self.data_path, f"duplicate_backup_dataset_{self.current_dataset_id}")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        total_size_saved = 0

        for file_name in files_to_backup:
            files_to_move = self._get_all_related_files(file_name)

            for file_path in files_to_move:
                if os.path.exists(file_path):  # Check if file exists before moving
                    rel_path = os.path.relpath(file_path, self.data_path)
                    backup_path = join(backup_dir, rel_path)

                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    total_size_saved += os.path.getsize(file_path)
                    shutil.move(file_path, backup_path)

        total_mb_saved = total_size_saved / (1024 * 1024)
        logger.info(f"Cleanup complete for dataset {self.current_dataset_id}. "
                    f"Moved {total_mb_saved:.2f} MB to {backup_dir}")

    def _load_and_preprocess_features(self, files_to_process: List[str], load_deepest_feature=False) -> Union[
        tuple[Tensor, list[Any], list[Any]], tuple[list[Any], list[Any]]]:
        """
        Load and preprocess features.
        """
        bottleneck_features = []
        valid_indices = []
        valid_filenames = []

        print("Loading features...")
        for idx, file_name in enumerate(tqdm(files_to_process)):
            # Load features
            feature_path = join(self.data_path, "feature_pkl", file_name[:-4] + ".pkl")
            if not self._verify_file_exists(feature_path):
                continue

            try:
                with open(feature_path, 'rb') as f:
                    features_dict = pickle.load(f)
                    # Get bottleneck features
                    bottleneck = features_dict[self.num_features - 1]
            except Exception as e:
                self.file_tracker.log_corrupted_file(feature_path, e)
                continue

            if load_deepest_feature:
                # Store bottleneck features
                bottleneck_features.append(bottleneck)
            valid_indices.append(idx)
            valid_filenames.append(file_name)

        # Process bottleneck features
        if load_deepest_feature:
            bottleneck_tensor = torch.from_numpy(np.stack(bottleneck_features)).float()
            orig_shape = bottleneck_tensor.shape
            bottleneck_flat = bottleneck_tensor.reshape(orig_shape[0], -1)
            return bottleneck_flat, valid_indices, valid_filenames

        return valid_indices, valid_filenames

    def __len__(self):
        return sum(len(buffer.valid_indices) for buffer in self.dataset_buffers.values())

    def __getitem__(self, index):
        # Find the correct dataset and local index
        current_count = 0
        for dataset_id, buffer in self.dataset_buffers.items():
            buffer_size = len(buffer.valid_indices)
            if current_count + buffer_size > index:
                local_index = buffer.valid_indices[index - current_count]
                file_name = buffer.data_patches[local_index]
                data_dict = self._load_item(file_name, dataset_id)
                # Add dataset_id to the returned dictionary
                data_dict['dataset_id'] = dataset_id
                return data_dict
            current_count += buffer_size
        raise IndexError("Index out of range")

    def _load_item(self, file_name: str, dataset_id: int):
        """Load a single item with proper error handling."""
        try:
            data_dict = {}

            # Load features and skips
            if self.load_skips:
                pkl_path = join(self.data_path, "feature_pkl", file_name[:-4] + ".pkl")
                if not self._verify_file_exists(pkl_path):
                    raise FileNotFoundError(f"Missing feature file: {pkl_path}")
                data_dict['features_and_skips'] = load_pickle(pkl_path)
            else:
                feature_path = join(
                    self.data_path,
                    "features",
                    file_name[:-4] + f"_{self.num_features - 1}.npy"
                )
                if not self._verify_file_exists(feature_path):
                    raise FileNotFoundError(f"Missing feature file: {feature_path}")
                data_dict['features_and_skips'] = self.constant_skips + [np.load(feature_path)]

            # Analyze sparsity of the first feature map
            first_feature_map = data_dict['features_and_skips'][0]
            self.sparsity_analyzer.update_stats(first_feature_map, dataset_id)

            # Load target based on type
            if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
                gt_path = join(self.data_path, "gt", file_name)
                if not self._verify_file_exists(gt_path):
                    raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
                gt_patch = np.load(gt_path)
                data_dict['target'] = gt_patch[None, None]
            elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
                pred_path = join(self.data_path, "predictions", file_name[:-4] + "_0.npy")
                if not self._verify_file_exists(pred_path):
                    raise FileNotFoundError(f"Missing prediction file: {pred_path}")
                gt_patch = np.load(pred_path)
                data_dict['target'] = gt_patch[None, None]

            return data_dict

        except Exception as e:
            logger.error(f"Error loading file {file_name} from dataset {dataset_id}: {str(e)}")
            raise


class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None,
                 num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0,
                 worker_init_fn=None, multiprocessing_context=None, generator=None, *,
                 prefetch_factor=None, persistent_workers: bool = False, pin_memory_device: str = "",
                 deep_supervision_scales=None,
                 deterministic=False):
        self.deep_supervision_scales = deep_supervision_scales

        assert len(dataset) >= batch_size, f"{len(dataset)}, {batch_size}"

        def my_collate_function(list_of_samples: list[dict]):
            # process the list_of_samples to create a batch and return it
            # each dict contains: 'features_and_skips', 'target'
            B = len(list_of_samples)
            output_batch = dict()

            # process targets
            # targets = []
            # for res in range(len(list_of_samples[0]['target'])):
            #    l = []
            #    for b in range(B):
            #        l.append(torch.from_numpy(list_of_samples[b]['target'][res]))
            #    targets.append(torch.vstack(l))
            # output_batch['target'] = targets

            if dataset.target_type in [FeatureRehearsalTargetType.GROUND_TRUTH,
                                       FeatureRehearsalTargetType.DISTILLED_OUTPUT]:
                targets = []
                for b in range(B):
                    targets.append(list_of_samples[b]['target'])
                targets = np.vstack(targets)
                output_batch['target'] = downsampling.downsample_seg_for_ds_transform2(targets,
                                                                                       self.deep_supervision_scales)
            elif dataset.target_type in [FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION]:
                assert False, "not implemented"
            elif dataset.target_type in [FeatureRehearsalTargetType.NONE]:
                pass
            else:
                assert False

            # process features_and_skips
            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['features_and_skips'][res]))
                features_and_skips.append(torch.vstack(l))
            output_batch['data'] = features_and_skips

            output_batch['dataset_id'] = torch.IntTensor([sample['dataset_id'] for sample in list_of_samples])

            if hasattr(dataset, 'load_meta') and dataset.load_meta:
                output_batch['slice_idx_normalized'] = torch.FloatTensor(
                    [sample['slice_idx_normalized'] for sample in list_of_samples])

            return output_batch

        # if sampler is None and shuffle is None or shuffle is True:
        #    sampler = RandomSampler(dataset, replacement=True, num_samples=5000), #<-- this is enough for 10 epochs but maybe this needs to be set higher (?)
        #    shuffle = None #<- sampler and shuffle are mutually exclusive. The random sampler already samples shuffled data, so this is fine.

        def seed_worker(worker_id):
            if deterministic:
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            if worker_init_fn != None:
                worker_init_fn(worker_id)

        super().__init__(dataset, batch_size, shuffle,
                         sampler,
                         batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, seed_worker,
                         multiprocessing_context, generator, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
    # TODO handle batch size
    # TODO handle foreground oversampling


class BalancedBatchSampler(Sampler):
    """
    Samples elements with balanced representation across datasets and domain-level foreground oversampling.
    For each domain's portion of the batch, implements nnUNet's sampling strategy:
    33.3% foreground patches, 66.7% random patches.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        """
        Args:
            dataset: The dataset (FeatureRehearsalDataset or FeatureRehearsalMultiDataset)
            batch_size: Size of each batch
            drop_last: If True, drops the last incomplete batch
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Handle both direct and wrapped datasets
        self.base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        self.num_copies = dataset.num_copies if hasattr(dataset, 'num_copies') else 1

        self.num_datasets = len(self.base_dataset.dataset_buffers)

        # Calculate base samples per dataset (before foreground/random split)
        self.samples_per_dataset = max(batch_size // self.num_datasets, 1)

        # For each dataset's sub-batch, calculate foreground vs random split
        self.foreground_per_dataset = max(self.samples_per_dataset // 3, 1)  # 33.3% with minimum 1
        self.random_per_dataset = self.samples_per_dataset - self.foreground_per_dataset

        # Calculate dataset sizes
        self.dataset_sizes = {
            dataset_id: len(buffer.valid_indices) * (int(self.num_copies) if self.num_copies > 1 else 1)
            for dataset_id, buffer in self.base_dataset.dataset_buffers.items()
        }

        # Calculate total batches
        min_possible_batches = min(
            size // self.samples_per_dataset
            for size in self.dataset_sizes.values()
        )
        self.total_batches = min_possible_batches if drop_last else min_possible_batches + 1

        # Initialize foreground indices mapping
        self.foreground_indices = self._get_foreground_indices()

        # Validate each dataset has enough foreground samples
        self._validate_foreground_samples()

    def _get_foreground_indices(self) -> Dict[int, List[int]]:
        """
        Create mapping of dataset IDs to indices containing foreground classes.
        Returns:
            Dict mapping dataset_id to list of indices with foreground content
        """
        foreground_indices = defaultdict(list)

        print("Analyzing patches for foreground content...")
        for dataset_id, buffer in self.base_dataset.dataset_buffers.items():
            dataset_start = sum(self.dataset_sizes[d] for d in range(dataset_id))

            for idx, file_name in enumerate(buffer.data_patches):
                if idx in buffer.valid_indices:
                    # Load ground truth
                    gt_path = os.path.join(self.base_dataset.data_path, "gt", file_name)
                    try:
                        gt = np.load(gt_path)
                        # Check if patch contains any foreground classes (any class > 0)
                        if np.any(gt > 0):
                            # For MultiDataset, create copies of the index
                            for copy in range(int(self.num_copies)):
                                global_idx = dataset_start + idx + (copy * len(buffer.valid_indices))
                                foreground_indices[dataset_id].append(global_idx)
                    except Exception as e:
                        logger.warning(f"Could not load GT for foreground analysis: {e}")
                        continue

        return dict(foreground_indices)

    def _validate_foreground_samples(self):
        """
        Validate that each dataset has enough foreground samples for at least one batch.
        Adjusts foreground/random split if necessary.
        """
        for dataset_id, foreground_indices in self.foreground_indices.items():
            num_foreground = len(foreground_indices)
            if num_foreground < self.foreground_per_dataset:
                logger.warning(
                    f"Dataset {dataset_id} has only {num_foreground} foreground samples, "
                    f"which is less than the required {self.foreground_per_dataset} per batch. "
                    "Adjusting foreground/random split for this dataset."
                )
                # Adjust split for this specific dataset
                self.foreground_per_dataset = min(self.foreground_per_dataset, num_foreground)
                self.random_per_dataset = self.samples_per_dataset - self.foreground_per_dataset

    def _get_dataset_indices(self, dataset_id: int) -> List[int]:
        """
        Get balanced foreground and random indices for a specific dataset.
        """
        indices = []
        dataset_start = sum(self.dataset_sizes[d] for d in range(dataset_id))
        dataset_size = self.dataset_sizes[dataset_id]

        # Get foreground indices for this dataset
        if dataset_id in self.foreground_indices:
            foreground_pool = self.foreground_indices[dataset_id]
            if len(foreground_pool) >= self.foreground_per_dataset:
                indices.extend(random.sample(foreground_pool, self.foreground_per_dataset))
            else:
                # Use all available foreground samples
                indices.extend(foreground_pool)
                # Make up the difference with random samples
                additional_random = self.foreground_per_dataset - len(foreground_pool)
                self.random_per_dataset += additional_random

        # Get random indices for this dataset
        all_indices = list(range(dataset_start, dataset_start + dataset_size))
        random_pool = list(set(all_indices) - set(indices))  # Exclude already selected foreground indices
        if len(random_pool) >= self.random_per_dataset:
            indices.extend(random.sample(random_pool, self.random_per_dataset))
        else:
            indices.extend(random_pool)

        random.shuffle(indices)
        return indices

    def __iter__(self):
        for _ in range(self.total_batches):
            batch_indices = []

            # Get indices for each dataset's portion of the batch
            for dataset_id in range(self.num_datasets):
                dataset_indices = self._get_dataset_indices(dataset_id)
                batch_indices.extend(dataset_indices)

            # Shuffle the complete batch
            random.shuffle(batch_indices)

            if len(batch_indices) > 0:  # Only yield non-empty batches
                yield from batch_indices[:self.batch_size]

    def __len__(self):
        if self.drop_last:
            return self.total_batches * self.batch_size
        # Calculate total samples including partial last batch
        total = 0
        for size in self.dataset_sizes.values():
            total += (size // self.samples_per_dataset) * self.samples_per_dataset
            if not self.drop_last:
                total += size % self.samples_per_dataset
        return total


class BalancedFeatureRehearsalDataLoader(FeatureRehearsalDataLoader):
    """DataLoader that ensures balanced sampling across different datasets."""

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor=None,
                 persistent_workers=False, pin_memory_device="", deep_supervision_scales=None,
                 deterministic=False):
        # Get the underlying FeatureRehearsalDataset if we're given a MultiDataset
        base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

        # Get dataset sizes for each domain
        dataset_sizes = {}
        for dataset_id, buffer in base_dataset.dataset_buffers.items():
            dataset_sizes[dataset_id] = len(buffer.valid_indices)

        # Create balanced sampler
        sampler = BalancedBatchSampler(dataset, batch_size, drop_last)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=None,  # Handled by sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            deep_supervision_scales=deep_supervision_scales,
            deterministic=deterministic
        )


class InfiniteIterator():
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.dataiter = iter(self.dataloader)

    def __next__(self):
        try:
            return next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            return next(self.dataiter)


class FeatureRehearsalConcatDataset(ConcatDataset):
    def __init__(self, main_dataset, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.dataset = main_dataset
        self.target_type = main_dataset.target_type

        self.load_meta = main_dataset.load_meta


class FeatureRehearsalMultiDataset(Dataset):
    # emulates multiple copies of the given dataset
    def __init__(self, dataset: FeatureRehearsalDataset, num_copies: float = 2) -> None:
        self.dataset = dataset
        self.target_type = dataset.target_type
        self.num_copies = num_copies
        self.load_meta = dataset.load_meta

    def __len__(self):
        return int(len(self.dataset) * self.num_copies)

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


class SparsityAnalyzer:
    def __init__(self):
        self.dataset_stats = {}

    def update_stats(self, feature_map, dataset_id):
        """Update running average statistics for a given feature map and dataset."""
        if dataset_id not in self.dataset_stats:
            self.dataset_stats[dataset_id] = {
                'avg_negative_ratio': 0.0,
                'avg_zero_ratio': 0.0,
                'num_samples': 0
            }

        stats = self.dataset_stats[dataset_id]

        # Compute current ratios
        total_elements = feature_map.size
        curr_negative_ratio = (feature_map < 0).sum().item() / total_elements
        curr_zero_ratio = (feature_map == 0).sum().item() / total_elements

        # Update running averages
        n = stats['num_samples']
        stats['avg_negative_ratio'] = (n * stats['avg_negative_ratio'] + curr_negative_ratio) / (n + 1)
        stats['avg_zero_ratio'] = (n * stats['avg_zero_ratio'] + curr_zero_ratio) / (n + 1)
        stats['num_samples'] += 1

    def get_summary(self):
        """Get summary statistics for all datasets."""
        summary = {}
        for dataset_id, stats in self.dataset_stats.items():
            summary[dataset_id] = {
                'sparsity_ratio': stats['avg_negative_ratio'] + stats['avg_zero_ratio'],
                'negative_ratio': stats['avg_negative_ratio'],
                'zero_ratio': stats['avg_zero_ratio'],
                'samples_analyzed': stats['num_samples']
            }
        return summary