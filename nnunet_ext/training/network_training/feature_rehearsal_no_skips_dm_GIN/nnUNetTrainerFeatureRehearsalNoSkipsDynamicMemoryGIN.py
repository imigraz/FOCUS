#########################################################################################################
# ------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

import math
import shutil

from nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet_ext.training.DataLoader2DWithSlices import DataLoader2DWithSlices
from nnunet_ext.training.network_training.feature_rehearsal2.nnUNetTrainerFeatureRehearsal2 import \
    nnUNetTrainerFeatureRehearsal2
from nnunet_ext.training.network_training.feature_rehearsal_no_skips_dm.nnUNetTrainerFeatureRehearsalNoSkipsDynamicMemory import \
    nnUNetTrainerFeatureRehearsalNoSkipsDynamicMemory
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
    FeatureRehearsalMultiDataset, FeatureSelectionMethod
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.utilities.helpful_functions import GINGroupConv2D

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

FEATURE_PATH = "extracted_features"


class nnUNetTrainerFeatureRehearsalNoSkipsDynamicMemoryGIN(nnUNetTrainerFeatureRehearsalNoSkipsDynamicMemory):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 use_progress=True,
                 identifier=default_plans_identifier, extension='feature_rehearsal_no_skips_dm_GIN',
                 tasks_list_with_char=None,
                 target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float = 1.0,
                 layer_name_for_feature_extraction: str = "",
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False,
                 seed: int = 42):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        target_type = target_type.DISTILLED_OUTPUT
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                         tasks_list_with_char,
                         target_type, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln,
                         do_LSA,
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

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #

        if not isinstance(self.network, Generic_UNet_no_skips):
            self.network.__class__ = Generic_UNet_no_skips

        rehearse = False

        if do_backprop and len(
                self.mh_network.heads.keys()) > 1:  # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            # probability_for_rehearsal = self.num_feature_rehearsal_cases / (self.num_feature_rehearsal_cases + len(self.dataset_tr))
            # v = torch.bernoulli(torch.tensor([probability_for_rehearsal]))[0]# <- unpack value {0,1}
            # rehearse = (v == 1)
            rehearse = self.rehearse
            self.rehearse = not self.rehearse

        if rehearse:
            try:
                data_dict = next(self.feature_rehearsal_dataiter)
            except StopIteration:
                self.feature_rehearsal_dataiter = iter(self.feature_rehearsal_dataloader)
                data_dict = next(self.feature_rehearsal_dataiter)
        else:
            data_dict = next(data_generator)

        data = data_dict['data']  # torch.Tensor (normal),    list[torch.Tensor] (rehearsal)
        target = data_dict['target']  # list[torch.Tensor]
        if not rehearse:
            properties = data_dict['properties']
            slice_indices = data_dict['slice_indices']
        else:
            properties = None

        # print(data_dict.keys())
        # print(data_dict['keys'])
        # print(data.shape)
        # print(type(target))
        # print(len(target))
        # for t in target:
        #    print(t.shape)
        # exit()

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if not rehearse and random.random() < 0.50:
            augmentor = GINGroupConv2D()
            data = augmentor(data)
            del augmentor

        self.optimizer.zero_grad()
        if self.fp16:
            with autocast():
                if rehearse:
                    output = self.network.feature_forward(data)
                else:
                    output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if rehearse:
                output = self.network.feature_forward(data)
            else:
                output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target, properties, slice_indices)

        del target

        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()

        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l
