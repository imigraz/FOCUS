#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

import math
import shutil

from pandas.core.common import random_state

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet_ext.training.network_training.feature_rehearsal2.nnUNetTrainerFeatureRehearsal2 import \
    nnUNetTrainerFeatureRehearsal2
from nnunet_ext.training.network_training.feature_rehearsal_no_skips.nnUNetTrainerFeatureRehearsalNoSkips import \
    nnUNetTrainerFeatureRehearsalNoSkips
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

from _warnings import warn
from typing import Tuple
import timeit
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from itertools import tee
import pickle
from nnunet_ext.network_architecture.generic_UNet import Generic_UNet
import random
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
import torch.nn.functional as F
import torch
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from multiprocessing import Process, Queue
from nnunet_ext.inference import predict
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.utils import pad_nd_image

from nnunet_ext.training.FeatureRehearsalDatasetDM import FeatureRehearsalDataset, BalancedFeatureRehearsalDataLoader, \
    FeatureRehearsalMultiDataset, FeatureSelectionMethod, FeatureRehearsalDataLoader
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


FEATURE_PATH = "extracted_features"


class nnUNetTrainerFeatureRehearsalNoSkipsDynamicMemory(nnUNetTrainerFeatureRehearsalNoSkips):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='feature_rehearsal_no_skips_dm', tasks_list_with_char=None,
                 target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float= 1.0,
                 layer_name_for_feature_extraction: str="",
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False, seed: int = 42):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         target_type, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA,
                         do_SPT, network, use_param_split, seed)

        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)

        self.dataset_dm = FeatureRehearsalDataset(
            join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time),
            self.deep_supervision_scales,
            self.target_type,
            num_features,
            load_skips=False,
            selection_method=FeatureSelectionMethod.RANDOM,
            retention_ratio=1.00,
            random_state=self.seed,
        )

    def store_features(self, task):
        self.print_to_log_file("extract features!")

        with torch.no_grad():
            # preprocess training cases and put them in a queue

            input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr')

            expected_num_modalities = load_pickle(self.plans_file)['num_modalities']  # self.plans
            case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

            ## take train cases only

            # print(case_ids)                 # <- all cases from this dataset
            # print(self.dataset_tr.keys())   # <- all train cases from this dataset
            assert set(self.dataset_tr.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(
                self.dataset_tr.keys()) + " " + str(case_ids)
            case_ids = list(self.dataset_tr.keys())

            ## take train cases subset
            case_ids = random.sample(case_ids, round(len(case_ids) * self.num_rehearsal_samples_in_perc))
            self.num_feature_rehearsal_cases += len(case_ids)

            self.print_to_log_file("the following cases will be used for feature rehearsal:" + str(case_ids))

            output_folder = join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time)
            maybe_mkdir_p(output_folder)

            output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
            all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                              len(i) == (len(j) + 12)] for j in case_ids]

            output_filenames = output_files[0::1]

            assert len(list_of_lists) == len(output_filenames)

            for o in output_filenames:
                dr, f = os.path.split(o)
                if len(dr) > 0:
                    maybe_mkdir_p(dr)

            ground_truth_segmentations = []
            for input_path_in_list in list_of_lists:
                input_path = input_path_in_list[0]
                # read segmentation file and place it in ground_truth_segmentations
                input_path_array = input_path.split('/')
                assert (input_path_array[-2] == "imagesTr")
                input_path_array[-2] = "labelsTr"
                assert (input_path_array[-1].endswith('_0000.nii.gz'))
                input_path_array[-1] = input_path_array[-1][:-12] + '.nii.gz'

                segmentation_path = join(*input_path_array)
                segmentation_path = "/" + segmentation_path
                ground_truth_segmentations.append(segmentation_path)

            print(ground_truth_segmentations)

            # print(list_of_lists[0::1])
            # exit()
            # preprocessing_generator = predict.preprocess_multithreaded(self, list_of_lists[0::1], output_filenames)
            preprocessing_generator = self._preprocess_multithreaded(list_of_lists[0::1], output_filenames,
                                                                     segs_from_prev_stage=ground_truth_segmentations)

            # for all preprocessed training cases with seg mask
            for preprocessed in preprocessing_generator:
                output_filename, (d, dct), gt_segmentation = preprocessed

                if isinstance(d, str):
                    assert isinstance(gt_segmentation, str)
                    data = np.load(d)
                    os.remove(d)
                    d = data

                    s = np.load(gt_segmentation)
                    os.remove(gt_segmentation)
                    gt_segmentation = s

                assert np.all(d.shape[1:] == gt_segmentation.shape), str(d.shape) + " " + str(gt_segmentation.shape)
                # unpack channel dimension on data

                # turn off deep supervision ???
                # step_size = 0.5 # TODO verify!!!
                # pad_border_mode = 'constant'
                # pad_kwargs = {'constant_values': 0}
                # mirror_axes = self.data_aug_params['mirror_axes']
                current_mode = self.network.training
                self.network.eval()

                ds = self.network.do_ds

                self.network.do_ds = False  # default: False    TODO
                do_mirroring = False  # default: True      only slight performance gain anyways
                mirror_axes = self.data_aug_params['mirror_axes']  # hard coded
                use_sliding_window = True  # hard coded
                step_size = 0.5  # default        (?)
                use_gaussian = True  # hard coded
                pad_border_mode = 'constant'  # default        (unset)
                pad_kwargs = None  # default        (unset)
                all_in_gpu = False  # default        (?)
                verbose = True  # default        (unset)
                mixed_precision = True  # default        (?)

                ret = self.network.predict_3D(d, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                              use_sliding_window=use_sliding_window, step_size=step_size,
                                              patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                              use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                              pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                              mixed_precision=mixed_precision,
                                              ground_truth_segmentation=gt_segmentation,
                                              feature_dir=output_filename[:-7],
                                              layer_name_for_feature_extraction=self.layer_name_for_feature_extraction)

                self.network.do_ds = ds
                self.network.train(current_mode)
            # END: for preprocessed

            ## update dataloader
            if not hasattr(self, 'feature_rehearsal_dataloader'):
                self.dataset_dm.update_features()
                dataset = FeatureRehearsalMultiDataset(self.dataset_dm,
                                                       math.ceil((self.batch_size * 250) / len(self.dataset_dm)))
                dataloader = BalancedFeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=0,
                                                                pin_memory=True,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                deterministic=self.deterministic)
                self.feature_rehearsal_dataloader = dataloader
                self.feature_rehearsal_dataiter = iter(dataloader)
            else:
                del self.feature_rehearsal_dataloader
                del self.feature_rehearsal_dataiter
                self.dataset_dm.update_features()
                dataset = FeatureRehearsalMultiDataset(self.dataset_dm,
                                                       math.ceil((self.batch_size * 250) / len(self.dataset_dm)))
                dataloader = BalancedFeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=0,
                                                                pin_memory=True,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                deterministic=self.deterministic)
                self.feature_rehearsal_dataloader = dataloader
                self.feature_rehearsal_dataiter = iter(dataloader)
                # self.tr_gen
                # TODO self.oversample_foreground_percent
                # https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample

    def clean_up(self):
        """Clean up all feature-related folders including backup folders"""
        base_path = join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time)

        # Clean main feature folders
        for folder in ["gt", "features", "predictions", "feature_pkl"]:
            path = join(base_path, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                # Remove all files and subdirectories
                for item in os.listdir(path):
                    item_path = join(path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            assert len(os.listdir(path)) == 0

        # Clean backup folders
        for item in os.listdir(base_path):
            if item.startswith('duplicate_backup_dataset_'):
                backup_path = join(base_path, item)
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                    print(f"Removed backup folder: {backup_path}")

        # Verify all backup folders are removed
        backup_folders = [f for f in os.listdir(base_path)
                          if f.startswith('duplicate_backup_dataset_')
                          and os.path.isdir(join(base_path, f))]
        assert len(backup_folders) == 0, "Not all backup folders were removed"
        print("Cleanup completed successfully")