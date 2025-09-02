from typing import Iterable
import torch, os, random
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset, Sampler
import numpy as np
from nnunet.training.data_augmentation import downsampling
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_pickle, join
from torch.utils.data.dataset import Dataset

class FeatureRehearsalTargetType(Enum):
    GROUND_TRUTH = 1
    DISTILLED_OUTPUT = 2
    DISTILLED_DEEP_SUPERVISION = 3
    NONE = 4

class FeatureRehearsalDataset(Dataset):
    def __init__(self, data_path: str, deep_supervision_scales: list[list[float]], target_type: FeatureRehearsalTargetType, num_features: int,
                 new_task_idx:int = None, old_dict_from_file_name_to_task_idx: dict = None, load_skips: bool = True, constant_skips: np.ndarray=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_patches = os.listdir(join(data_path, "gt"))
        self.data_patches.sort()
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features
        self.load_skips = load_skips
        self.load_meta = False

        if not self.load_skips:
            if constant_skips is None:
                assert len(self.data_patches) > 0
                self.constant_skips = [np.zeros_like(f) for f in load_pickle(join(self.data_path, "feature_pkl", self.data_patches[0][:-4] + ".pkl"))[:-1] ]
            else:
                self.constant_skips = constant_skips

        self.store_task_idx = new_task_idx is not None or old_dict_from_file_name_to_task_idx is not None
        if self.store_task_idx:
            assert old_dict_from_file_name_to_task_idx is not None
            self.task_idx_array = []
            for file in self.data_patches:
                if file in old_dict_from_file_name_to_task_idx.keys():
                    self.task_idx_array.append(old_dict_from_file_name_to_task_idx[file])
                else:
                    assert new_task_idx is not None, file
                    self.task_idx_array.append(new_task_idx)

    def features_to_features_and_skips(self, features):
        assert not self.load_skips
        if not isinstance(features, list):
            features = [features]
        return self.constant_skips + features

    def get_dict_from_file_name_to_task_idx(self):
        assert self.store_task_idx
        d ={}
        for i, file in enumerate(self.data_patches):
            d[file] = self.task_idx_array[i]
        return d

    def __len__(self):
        return len(self.data_patches)
    
    def __getitem__(self, index):

        data_dict = dict()
        if self.load_skips:
            data_dict['features_and_skips'] = load_pickle(join(self.data_path, "feature_pkl", self.data_patches[index][:-4] + ".pkl"))[:-2]
        else:
            data_dict['features_and_skips'] = self.constant_skips + [np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(self.num_features-1) +".npy"))]
        #for i in range(self.num_features):
        #    data_dict['features_and_skips'].append(np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(i) +".npy")))
        
        if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            gt_patch = np.load(join(self.data_path, "gt",self.data_patches[index]))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
            gt_patch = np.load(join(self.data_path, "predictions", self.data_patches[index][:-4] + "_" + str(0) +".npy"))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION:
            assert False, "not implemented yet"
        elif self.target_type == FeatureRehearsalTargetType.NONE:
            pass
        else:
            assert False

        if self.store_task_idx:
            data_dict['task_idx'] = self.task_idx_array[index]

        return data_dict
    
class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size = 1, shuffle = None, sampler= None, batch_sampler= None, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn = None, multiprocessing_context=None, generator=None, *, 
                 prefetch_factor = None, persistent_workers: bool = False, pin_memory_device: str = "", deep_supervision_scales=None,
                 deterministic=False):
        self.deep_supervision_scales = deep_supervision_scales

        assert len(dataset) >= batch_size, f"{len(dataset)}, {batch_size}"

        def my_collate_function(list_of_samples: list[dict]):
            #process the list_of_samples to create a batch and return it
            # each dict contains: 'features_and_skips', 'target'
            B = len(list_of_samples)
            output_batch = dict()

            #process targets
            #targets = []
            #for res in range(len(list_of_samples[0]['target'])):
            #    l = []
            #    for b in range(B):
            #        l.append(torch.from_numpy(list_of_samples[b]['target'][res]))
            #    targets.append(torch.vstack(l))
            #output_batch['target'] = targets

            if dataset.target_type in [FeatureRehearsalTargetType.GROUND_TRUTH, FeatureRehearsalTargetType.DISTILLED_OUTPUT]:
                targets = []
                for b in range(B):
                    targets.append(list_of_samples[b]['target'])
                targets = np.vstack(targets)
                output_batch['target'] = downsampling.downsample_seg_for_ds_transform2(targets, self.deep_supervision_scales)
            elif dataset.target_type in [FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION]:
                assert False, "not implemented"
            elif dataset.target_type in [FeatureRehearsalTargetType.NONE]:
                pass
            else:
                assert False

            #process features_and_skips
            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['features_and_skips'][res]))
                features_and_skips.append(torch.vstack(l))
            output_batch['data'] = features_and_skips

            if dataset.store_task_idx:
                output_batch['task_idx'] = torch.IntTensor([sample['task_idx'] for sample in list_of_samples])

            if hasattr(dataset, 'load_meta') and dataset.load_meta:
                output_batch['slice_idx_normalized'] = torch.FloatTensor([sample['slice_idx_normalized'] for sample in list_of_samples])

            return output_batch
        
        #if sampler is None and shuffle is None or shuffle is True:
        #    sampler = RandomSampler(dataset, replacement=True, num_samples=5000), #<-- this is enough for 10 epochs but maybe this needs to be set higher (?)
        #    shuffle = None #<- sampler and shuffle are mutually exclusive. The random sampler already samples shuffled data, so this is fine.


        def seed_worker(worker_id):
            if deterministic:
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            if worker_init_fn != None:
                worker_init_fn(worker_id)

        super().__init__(dataset, batch_size, shuffle, 
                         sampler,
                         batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, seed_worker, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
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
        self.store_task_idx = main_dataset.store_task_idx
        self.target_type = main_dataset.target_type
        
        self.load_meta = main_dataset.load_meta

class FeatureRehearsalMultiDataset(Dataset):
    #emulates multiple copies of the given dataset
    def __init__(self, dataset: FeatureRehearsalDataset, num_copies: float=2) -> None:
        self.dataset = dataset
        self.store_task_idx = dataset.store_task_idx
        self.target_type = dataset.target_type
        self.num_copies = num_copies
        self.load_meta = dataset.load_meta

    def __len__(self):
        return int(len(self.dataset) * self.num_copies)
    
    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]