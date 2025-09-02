###############################################################################################################
#----------This class represents a Generic ViT_U-Net model based on the ViT and nnU-Net architecture----------#
###############################################################################################################

# Changes to main:
# - Use own Generic_UNet_ class
# - Enhance forward method to extract features during training
# - Implement feature forward method that trains using features and skip connections

from ast import Tuple
from torch import nn
from nnunet_ext.utilities.helpful_functions import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet_ext.network_architecture.GenericUNet_ import Generic_UNet_

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List

from torch.cuda.amp import autocast

class Generic_UNet(Generic_UNet_):
    r"""This class is a Module that can be used for any segmentation task. It represents a generic combination of the
        Vision Transformer (https://arxiv.org/pdf/2010.11929.pdf) and the generic U-Net architecture known as the
        nnU-Net Framework.
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False,
                 vit_version='V1', vit_type='base', split_gpu=False, ViT_task_specific_ln=False, first_task_name=None,
                 do_LSA=False, do_SPT=False, FeatScale=False, AttnScale=False, useFFT=False, fourier_mapping=False,
                 f_map_type='none', conv_smooth=None, ts_msa=False, cross_attn=False, cbam=False, registration=None,
                 no_skips=False):
        r"""Generic U-Net with updated Encoder Decoder order"""
        
        # -- Initialize using parent class --> gives us a generic U-Net we need to alter to create our combined architecture -- #
        super(Generic_UNet, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                               feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                               final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features,
                                               basic_block, seg_output_use_bias, no_skips=no_skips)
        
        # -- Create copies of the different parts and delete them all again -- #
        conv_blocks_localization = self.conv_blocks_localization
        conv_blocks_context = self.conv_blocks_context
        td = self.td
        tu = self.tu
        seg_outputs = self.seg_outputs
        del self.conv_blocks_localization, self.conv_blocks_context, self.td, self.tu, self.seg_outputs

        # -- Re-register all modules properly using backups to create a specific order -- #
        # -- NEW Order: Encoder -- Decoder -- Segmentation Head
        self.conv_blocks_context = conv_blocks_context  # Encoder part 1
        self.td = td  # Encoder part 2
        self.tu = tu   # Decoder part 1
        self.conv_blocks_localization = conv_blocks_localization   # Decoder part 2
        self.seg_outputs = seg_outputs  # Segmentation head

    def forward(self, x, layer_name_for_feature_extraction: str = None):
        skips = []
        seg_outputs = []

        # Encoder path
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                features_and_skips = skips + [x]

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

            if layer_name_for_feature_extraction == "td." + str(d):
                features_and_skips = skips + [x]

        x = self.conv_blocks_context[-1](x)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            features_and_skips = skips + [x]

        # Store intermediate feature maps for last two layers
        last_features = []
        second_last_features = []

        # Decoder path
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)

            if layer_name_for_feature_extraction == "tu." + str(u):
                features_and_skips = skips + [x]

            x = self.conv_blocks_localization[u](x)

            # Store last two feature maps
            if u == len(self.tu) - 2:
                second_last_features = x
            elif u == len(self.tu) - 1:
                last_features = x

            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                features_and_skips = skips + [x]

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if layer_name_for_feature_extraction is not None:
            assert features_and_skips is not None and isinstance(features_and_skips, list)
            features_and_skips.extend([second_last_features, last_features])

        if self._deep_supervision and self.do_ds:
            if layer_name_for_feature_extraction is not None:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                  zip(list(self.upscale_logits_ops)[::-1],
                                                      seg_outputs[:-1][::-1])]), features_and_skips
            else:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                                  zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            if layer_name_for_feature_extraction is not None:
                return seg_outputs[-1], features_and_skips
            else:
                return seg_outputs[-1]

    def _freeze_conv_weights_in_module(self, module):
        """Helper function to freeze only convolutional weights in a module"""
        for name, param in module.named_parameters():
            # Freeze convolutional and linear layer weights/biases, but keep norm parameters trainable
            if any(layer_type in name.lower() for layer_type in ['conv', 'linear']):
                param.requires_grad_(False)
            # Keep normalization parameters trainable (batch_norm, instance_norm, group_norm, layer_norm)
            elif any(norm_type in name.lower() for norm_type in ['norm', 'bn', 'in', 'gn', 'ln']):
                param.requires_grad_(True)

    def _keep_norm_layers_in_train_mode(self, module):
        """Helper function to ensure normalization layers remain in training mode"""
        for submodule in module.modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                      nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                      nn.GroupNorm, nn.LayerNorm)):
                submodule.train(True)

    def freeze_conv_weights_only(self, layer_name_for_feature_extraction: str):
        """
        Freeze only convolutional weights while keeping normalization parameters trainable.
        This allows norm statistics to continue updating during training.
        """
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction

        # Process encoder context blocks
        for d in range(len(self.conv_blocks_context) - 1):
            self._freeze_conv_weights_in_module(self.conv_blocks_context[d])

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return

            if not self.convolutional_pooling:
                self._freeze_conv_weights_in_module(self.td[d])

            if layer_name_for_feature_extraction == "td." + str(d):
                return

        # Process final encoder block
        self._freeze_conv_weights_in_module(self.conv_blocks_context[-1])
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return

        # Process decoder blocks
        for u in range(len(self.tu)):
            self._freeze_conv_weights_in_module(self.tu[u])
            if layer_name_for_feature_extraction == "tu." + str(u):
                return

            self._freeze_conv_weights_in_module(self.conv_blocks_localization[u])
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return

        raise AssertionError(f"Invalid layer name for feature extraction: {layer_name_for_feature_extraction}")

    def get_frozen_parameter_info(self):
        """
        Utility method to check which parameters are frozen and which are trainable.
        Useful for debugging and verification.
        """
        frozen_params = []
        trainable_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        print(f"Frozen parameters ({len(frozen_params)}):")
        for name in frozen_params:
            print(f"  - {name}")

        print(f"\nTrainable parameters ({len(trainable_params)}):")
        for name in trainable_params:
            print(f"  - {name}")

        return frozen_params, trainable_params

    def train(self, train_mode: bool = True):
        """
        Override train method to keep normalization layers in training mode even when other parts are frozen.
        This ensures that normalization statistics continue to update.
        """
        super().train(train_mode)

        if not hasattr(self, 'layer_name_for_feature_extraction'):
            return self

        # Keep all normalization layers in training mode to allow statistics updates
        layer_name_for_feature_extraction = self.layer_name_for_feature_extraction

        # Process encoder context blocks
        for d in range(len(self.conv_blocks_context) - 1):
            # Keep norm layers in training mode
            self._keep_norm_layers_in_train_mode(self.conv_blocks_context[d])

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return self

            if not self.convolutional_pooling:
                self._keep_norm_layers_in_train_mode(self.td[d])

            if layer_name_for_feature_extraction == "td." + str(d):
                return self

        # Process final encoder block
        self._keep_norm_layers_in_train_mode(self.conv_blocks_context[-1])
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return self

        # Process decoder blocks
        for u in range(len(self.tu)):
            self._keep_norm_layers_in_train_mode(self.tu[u])
            if layer_name_for_feature_extraction == "tu." + str(u):
                return self

            self._keep_norm_layers_in_train_mode(self.conv_blocks_localization[u])
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return self

        return self


    def freeze_layers(self, layer_name_for_feature_extraction: str):
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        self.train(self.training)
        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return
            
            if not self.convolutional_pooling:
                self.td[d].requires_grad_(requires_grad=False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return
            
        self.conv_blocks_context[-1].requires_grad_(requires_grad=False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context)-1):
            return
        
        for u in range(len(self.tu)):
            self.tu[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return
            
            self.conv_blocks_localization[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        assert False, "we cannot end up here. maybe the layer name for feature extraction is wrong " + str(layer_name_for_feature_extraction)
        

    def feature_forward(self, features_and_skips: list[torch.Tensor]):
        #Attention: If deep supervision is activate, the output might contains less entries than you would expect!!!

        assert hasattr(self, 'layer_name_for_feature_extraction')
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        x = features_and_skips[-1]
        skips = features_and_skips[:-1]
        seg_outputs = []

        if layer == "conv_blocks_context":
            if id<len(self.conv_blocks_context)-1:
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[id](x)
            for d in range(id+1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)
        elif layer == "td":
            for d in range(id+1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)

        #for s in skips:
        #    print(s.shape)

        if id < len(self.conv_blocks_context)-1 and layer in ["td", "conv_blocks_context"]:
            x = self.conv_blocks_context[-1](x)
        
        if layer in ["td", "conv_blocks_context"]:  #in this case there is nothing to be done
            for u in range(len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                
                x = self.conv_blocks_localization[u](x)

                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        elif layer == "conv_blocks_localization":
            for u in range(id+1, len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu)-1:    # that means we only train the highest resolution segmentation head
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))

        else:
            assert layer == "tu"
            if id < len(self.conv_blocks_localization):
                x = self.conv_blocks_localization[id](x)
            for u in range(id+1, len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu)-1:    # that means we only train the highest resolution segmentation head and conv_blocks_localization[id] (see few lines prior)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))


        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                            zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]