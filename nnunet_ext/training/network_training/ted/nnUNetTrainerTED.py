from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential

# Based on https://github.com/PerceptionComputingLab/TED_DCMIS

class LossAKT(nn.Module):
    """Loss function for Adaptive Knowledge Transfer"""

    def __init__(self, gugf=True):
        super().__init__()
        self.epsilon = 1e-5
        self.uncertainty = gugf

    def forward(self, output, target):
        probabilities_old = target
        target = target.argmax(dim=1, keepdim=True)  # pseudo label

        probabilities = output
        log_probs = torch.log(probabilities + self.epsilon)
        log_probs_target = log_probs.gather(dim=1, index=target.long())

        net_predictions = log_probs.argmax(dim=1, keepdim=True)
        error_predictions = (net_predictions != target).float().detach()
        error_predictions_num = torch.sum(error_predictions)

        if self.uncertainty:
            jointly = probabilities_old * probabilities
            uncertainty = (torch.ones([1]).to(output.device) - torch.sum(jointly, dim=1)).detach()
        else:
            uncertainty = torch.ones([1]).to(output.device)

        loss = -torch.sum(uncertainty * error_predictions * log_probs_target) / (error_predictions_num + 1)
        return loss


class TEDLoss(nn.Module):
    def __init__(self, base_loss, weight_distill=0.001, gugf=True):
        """Combined loss for TED training with deep supervision support
        Args:
            base_loss: existing nnUNet loss with deep supervision
            weight_distill: weight for distillation losses
            gugf: whether to use uncertainty-guided unified filtering
        """
        super().__init__()
        self.base_loss = base_loss  # Use existing loss from parent class
        self.weight_distill = weight_distill
        self.akt_loss = LossAKT(gugf=gugf)
        self.epsilon = 1e-5

    def forward(self, net_output, old_outputs, gt_target, noise_outputs):
        """
        Args:
            net_output: list of network outputs for each deep supervision level
            target: list of (list of ground truth tensors, list of old model outputs) or just list of ground truth tensors
            noise_outputs: tuple of (list of new noise outputs, list of old noise outputs) for SKA
        """

        # Base segmentation loss using original nnUNet loss
        seg_loss = self.base_loss(net_output, gt_target)

        # If no old model output, return only segmentation loss
        if old_outputs is None:
            return seg_loss

        # Initialize distillation losses
        total_distill_loss = 0.0

        # Get deep supervision weights from base loss
        ds_loss_weights = self.base_loss.weight_factors if hasattr(self.base_loss, 'weight_factors') else [1.0] * len(net_output)

        # Unpack noise outputs
        new_noise_outputs, old_noise_outputs = noise_outputs

        # Calculate distillation losses for each deep supervision level
        for output, old_output, new_noise, old_noise, weight in zip(
                net_output, old_outputs, new_noise_outputs, old_noise_outputs, ds_loss_weights
        ):
            if weight == 0:
                continue

            output_softmax = softmax_helper(output)
            old_output_softmax = softmax_helper(old_output)

            # Image distillation loss
            distill_image = self.cross_entropy_with_probs(output_softmax, old_output_softmax)

            # AKT loss
            distill_akt = self.akt_loss(output_softmax, old_output_softmax)

            # SKA (noise) loss
            new_noise_softmax = softmax_helper(new_noise)
            old_noise_softmax = softmax_helper(old_noise)
            distill_noise = self.cross_entropy_with_probs(new_noise_softmax, old_noise_softmax)

            # Add weighted distillation loss for this level and normalize it the same way as nnUNet loss
            total_distill_loss += weight * (distill_image + distill_akt + distill_noise)

        # Combined loss with deep supervision
        return seg_loss + self.weight_distill * total_distill_loss

    def cross_entropy_with_probs(self, prediction, target):
        """Computes cross entropy between predictions and targets, both in probability space"""
        return (-(target * torch.log(prediction + self.epsilon)).sum(dim=1)).mean()


class nnUNetTrainerTED(nnUNetTrainerSequential):
    def __init__(self, *args, lambda_d=0.001, gugf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_d = lambda_d
        self.gugf = gugf

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None,
                   call_for_eval=False):
        """Initialize the trainer. Super class will set self.loss to DC_and_CE_loss"""
        # Initialize with parent class
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # Initialize TED-specific loss using deep supervision weights
        if training:
            self.loss = TEDLoss(
                base_loss=self.loss,  # Use existing loss directly
                weight_distill=self.lambda_d,
                gugf=self.gugf
            )

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        properties = data_dict['properties']
        slice_indices = data_dict['slice_indices']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                old_output = None
                noise_outputs = None

                # Get output from old model if available - use self.network_old from parent class
                if hasattr(self, 'network_old') and self.network_old is not None:
                    with torch.no_grad():
                        old_output = self.network_old(data)

                    noise = torch.randn_like(data).to(data.device)
                    noise_output = self.network(noise)
                    with torch.no_grad():
                        old_noise_output = self.network_old(noise)
                    noise_outputs = (noise_output, old_noise_output)

                # Calculate loss
                if not no_loss:
                    l = self.loss(output, old_output, target, noise_outputs)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            old_output = None
            noise_outputs = None

            # Get output from old model if available - use self.network_old from parent class
            if hasattr(self, 'network_old') and self.network_old is not None:
                with torch.no_grad():
                    old_output = self.network_old(data)

                noise = torch.randn_like(data).to(data.device)
                noise_output = self.network(noise)
                with torch.no_grad():
                    old_noise_output = self.network_old(noise)
                noise_outputs = (noise_output, old_noise_output)

            # Calculate loss
            if not no_loss:
                l = self.loss(output, old_output, target, noise_outputs)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target,
                                       properties, slice_indices)

        # Update the Multi Head Network after iteration
        if do_backprop:
            self.mh_network.update_after_iteration()

        if not no_loss:
            return l.detach().cpu().numpy() if detach else l