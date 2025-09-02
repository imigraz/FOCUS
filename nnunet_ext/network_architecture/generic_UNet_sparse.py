from ast import Tuple

from matplotlib import pyplot as plt
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


class Generic_UNet_sparse(Generic_UNet_):
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
        super(Generic_UNet_sparse, self).__init__(input_channels, base_num_features, num_classes, num_pool,
                                                  num_conv_per_stage,
                                                  feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs,
                                                  dropout_op,
                                                  dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision,
                                                  dropout_in_localization,
                                                  final_nonlin, weightInitializer, pool_op_kernel_sizes,
                                                  conv_kernel_sizes,
                                                  upscale_logits, convolutional_pooling, convolutional_upsampling,
                                                  max_num_features,
                                                  basic_block, seg_output_use_bias, no_skips=no_skips)

        conv_blocks_localization = self.conv_blocks_localization
        conv_blocks_context = self.conv_blocks_context
        td = self.td
        tu = self.tu
        seg_outputs = self.seg_outputs
        del self.conv_blocks_localization, self.conv_blocks_context, self.td, self.tu, self.seg_outputs

        self.conv_blocks_context = conv_blocks_context
        self.td = td
        self.tu = tu
        self.conv_blocks_localization = conv_blocks_localization
        self.seg_outputs = seg_outputs
        self.dropout_p = 0.9
        self.plot_counter = 0

    def enhance_contrast(self, image, percentile_low=2, percentile_high=98, window_width=None, window_level=None):
        """
        MRI-specific contrast enhancement using window/level adjustment
        """
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        if window_width is None or window_level is None:
            p_low = np.percentile(image, percentile_low)
            p_high = np.percentile(image, percentile_high)
            window_width = p_high - p_low
            window_level = (p_high + p_low) / 2

        min_value = window_level - window_width / 2
        max_value = window_level + window_width / 2
        return np.clip((image - min_value) / (max_value - min_value), 0, 1)

    def visualize_feature_rehearsal(self, img, features, spatial_mask):
        timestamp = int(time.time())

        # Add this before the zero counting
        unique_vals, counts = torch.unique(spatial_mask[0, 0], return_counts=True)
        print(f"Unique values in mask: {unique_vals.tolist()}")
        print(f"Counts for each value: {counts.tolist()}")
        total_pixels = spatial_mask[0, 0].numel()
        print(f"Percentage of non-zeros: {(counts[1] / total_pixels) * 100:.1f}%")

        zeros = (spatial_mask[0, 0] == 0).sum().item()
        total = spatial_mask[0, 0].numel()
        actual_dropout = (zeros / total) * 100

        print(f"Zero count: {zeros}")
        print(f"Total elements: {total}")
        print(f"Raw mask values unique: {torch.unique(spatial_mask[0, 0])}")

        for ch in range(features[0].shape[0]):
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            img_np = img[0].permute(1, 2, 0).detach().cpu()
            img_np = self.enhance_contrast(img_np[:, :, 0])
            feat_np = self.enhance_contrast(features[0, ch].detach().cpu())
            mask_np = spatial_mask[0, 0].detach().cpu().numpy()
            feat_dropout_np = feat_np * mask_np
            axs[0].imshow(img_np, cmap='gray', interpolation='nearest')
            axs[1].imshow(feat_np, cmap='gray', interpolation='nearest')
            axs[2].imshow(spatial_mask[0, 0].detach().cpu(), cmap='gray', interpolation='nearest')
            axs[3].imshow(feat_dropout_np, cmap='gray', interpolation='nearest')

            titles = ['Input', '1st Level Features', f'Dropout Mask ({actual_dropout:.1f}% zeros)', 'Sparse Features']
            for ax, title in zip(axs, titles):
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join('plots', f'privacy_features_ch{ch}_{timestamp}.png'))
            plt.close()

    def sparsify_features(self, x: torch.Tensor, during_normal_training: bool = False,
                          input: torch.Tensor = None) -> torch.Tensor:
        if self.dropout_p is None:
            self.dropout_p = 0.9
        if during_normal_training:
            batch_size = x.shape[0]
            num_to_dropout = torch.div(batch_size, 4, rounding_mode='floor')
            num_to_dropout = torch.maximum(torch.tensor(1), num_to_dropout)

            batch_indices = torch.randperm(batch_size, device=x.device)
            batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            batch_mask.index_fill_(0, batch_indices[:num_to_dropout], True)

            # Keep sampling until we get a non-zero mask
            spatial_mask = torch.zeros(batch_size, 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            while torch.all(spatial_mask == 0):
                spatial_mask = torch.bernoulli(
                    torch.ones(batch_size, 1, x.shape[2], x.shape[3],
                               device=x.device, dtype=x.dtype) * (1 - self.dropout_p)
                ) / (1 - self.dropout_p)

            spatial_mask = spatial_mask.expand(-1, x.shape[1], -1, -1)
            mask_expanded = batch_mask.view(-1, 1, 1, 1).expand_as(x)
            x_out = torch.where(mask_expanded, x * spatial_mask, x)

            return x_out
        else:
            # Keep sampling until we get a non-zero mask
            spatial_mask = torch.zeros_like(x[:, :1, :, :])
            while torch.all(spatial_mask == 0):
                spatial_mask = torch.bernoulli(torch.ones_like(x[:, :1, :, :]) * (1 - self.dropout_p)) / (
                        1 - self.dropout_p)

            spatial_mask = spatial_mask.expand(-1, x.shape[1], -1, -1)
            if input is not None:
                self.visualize_feature_rehearsal(input, x, spatial_mask)
            return x * spatial_mask

    def set_dropout_p(self, p: float):
        p = torch.clamp(torch.tensor(p), 0.0, 1.0).item()
        self.dropout_p = p

    def forward(self, x, layer_name_for_feature_extraction: str = None):
        skips = []
        seg_outputs = []

        # if you provide the input here you can use sparsify features for visualization
        original_input = None
        # original_input = x.clone()

        # Encoder path
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            if self.training:
                # During normal training, apply dropout to 25% of samples
                x = self.sparsify_features(x, during_normal_training=True)

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

            if layer_name_for_feature_extraction == "td." + str(d):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

        x = self.conv_blocks_context[-1](x)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            # for visualization of first feature maps, before and after dropout
            if original_input is not None:
                self.sparsify_features(skips[0], False, original_input)

            features_and_skips = [self.sparsify_features(skip) for skip in skips]
            features_and_skips.append(self.sparsify_features(x))

        # Decoder path
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)

            if layer_name_for_feature_extraction == "tu." + str(u):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            x = self.conv_blocks_localization[u](x)

            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                features_and_skips = [self.sparsify_features(skip) for skip in skips]
                features_and_skips.append(self.sparsify_features(x))

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

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

    def train(self, train_mode: bool = True):
        super().train(train_mode)
        if not hasattr(self, 'layer_name_for_feature_extraction'):
            return

        layer_name_for_feature_extraction = self.layer_name_for_feature_extraction

        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return

            if not self.convolutional_pooling:
                self.td[d].train(False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return

        self.conv_blocks_context[-1].train(False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return

        for u in range(len(self.tu)):
            self.tu[u].train(False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return

            self.conv_blocks_localization[u].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        assert False

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
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context) - 1):
            return

        for u in range(len(self.tu)):
            self.tu[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return

            self.conv_blocks_localization[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        assert False, "we cannot end up here. maybe the layer name for feature extraction is wrong " + str(
            layer_name_for_feature_extraction)

    def feature_forward(self, features_and_skips: list[torch.Tensor]):
        assert hasattr(self, 'layer_name_for_feature_extraction')
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        # Only use the first feature map
        x = features_and_skips[0]
        skips = []
        seg_outputs = []

        # Since we start with the first feature map (already processed by first conv block)
        # We start from index 1 in the encoder path
        if layer == "conv_blocks_context":
            # First feature map is already processed, so start from next block
            skips.append(x)  # Save the first feature map we received
            if not self.convolutional_pooling:
                x = self.td[0](x)

            for d in range(1, id + 1):
                x = self.conv_blocks_context[d](x)
                if d < id:
                    skips.append(x)
                    if not self.convolutional_pooling:
                        x = self.td[d](x)

            if id < len(self.conv_blocks_context) - 1:
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[id](x)

                # Continue with remaining encoder layers
                for d in range(id + 1, len(self.conv_blocks_context) - 1):
                    x = self.conv_blocks_context[d](x)
                    skips.append(x)
                    if not self.convolutional_pooling:
                        x = self.td[d](x)

        elif layer == "td":
            # First feature map is already processed by first conv block
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[0](x)

            for d in range(1, id + 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling and d < id:
                    x = self.td[d](x)

            # Continue with remaining encoder layers
            for d in range(id + 1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)

        # Process bottleneck if needed
        if id < len(self.conv_blocks_context) - 1 and layer in ["td", "conv_blocks_context"]:
            x = self.conv_blocks_context[-1](x)

        # Decoder path
        if layer in ["td", "conv_blocks_context"]:
            for u in range(len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        elif layer == "conv_blocks_localization":
            # Compute features up to the specified localization block
            for u in range(id):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)

            # Continue with remaining decoder layers
            for u in range(id, len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        else:  # layer == "tu"
            # Compute features up to the specified upsampling block
            for u in range(id):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)

            if id < len(self.conv_blocks_localization):
                x = self.conv_blocks_localization[id](x)

            # Continue with remaining decoder layers
            for u in range(id + 1, len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu) - 1:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]