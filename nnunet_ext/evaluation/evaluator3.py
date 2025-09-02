import os
import glob
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.inference.predict import predict_from_folder
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir, preprocessing_output_dir
from nnunet_ext.utilities.helpful_functions import get_ViT_LSA_SPT_folder_name, join_texts_with_char
from nnunet_ext.utilities.helpful_functions import dumpDataFrameToCsv, nestedDictToFlatTable
from batchgenerators.utilities.file_and_folder_operations import *


def find_experiment_folders(network: str, network_trainer: str, tasks_joined_name: str,
                            model_joined_name: str, plans_identifier: str) -> List[str]:
    """
    Find all experiment folders for a given trainer configuration.
    Supports folders with datetime stamps and experiment names (e.g., '26.11.2024_10:27:27_focus').
    """
    base_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name)
    pattern = f"{network_trainer}__*{plans_identifier}*"
    folders = glob.glob(join(base_path, pattern))
    return folders


def get_plans_file(trainer_path: str) -> str:
    """
    Get the plans.pkl file path, which is inside the SEQ folder.

    Args:
        trainer_path: Path to the trainer folder containing SEQ folder

    Returns:
        Path to the plans.pkl file
    """
    plans_path = join(trainer_path, "plans.pkl")
    if not os.path.exists(plans_path):
        raise FileNotFoundError(f"plans.pkl not found at {plans_path}")
    return plans_path


def build_trainer_and_output_path(network: str, network_trainer: str, trainer_path: str, tasks_joined_name: str,
                                model_joined_name: str, plans_identifier: str, transfer_heads: bool,
                                folder_n: str, use_vit: bool, ViT_task_specific_ln: bool,
                                vit_type: str, version: str, do_pod: bool, use_head: str,
                                fold: int, evaluate_on: str) -> Tuple[str, str]:
    """
    Build paths for trainer and output, supporting datetime experiment folders.
    Example trainer path structure:
    .../Task111_Prostate-BIDMC/nnUNetTrainerFOCUS__nnUNetPlansv2.1_26.11.2024_10:27:27_focus/Generic_UNet/SEQ/...
    """
    # Extract experiment folder name (including datetime stamp)
    trainer_parts = trainer_path.split(os.path.sep)
    experiment_folder = next(part for part in trainer_parts
                           if part.startswith(f"{network_trainer}__") and plans_identifier in part)

    # Build base path components for both trainer and output
    if use_vit:
        network_path = join(Generic_ViT_UNet.__name__ + "V" + version, vit_type,
                          'task_specific' if ViT_task_specific_ln else 'not_task_specific',
                          folder_n)
    else:
        network_path = Generic_UNet.__name__

    sequence_path = 'SEQ' if transfer_heads else 'MH'

    # Build complete trainer path
    complete_trainer_path = join(trainer_path, network_path, sequence_path)

    # Build output path
    output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name,
                      experiment_folder, network_path, sequence_path)

    if 'OwnM' in experiment_folder:
        output_path = join(os.path.sep, *output_path.split(os.path.sep)[:-1],
                         'pod' if do_pod else 'no_pod')

    output_path = join(output_path, f'head_{use_head}', f'fold_{fold}', f'Preds_{evaluate_on}')

    return complete_trainer_path, output_path


# Adapted from MetricsReloaded by Carole Sudre
# Source: https://github.com/MetricsReloaded/MetricsReloaded
# Licensed under Apache 2.0
import numpy as np
from scipy import ndimage


class MorphologyOps:
    """
    Taken directly from MetricsReloaded code.
    """
    def __init__(self, binary_img, connectivity=1):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.connectivity = connectivity

    def border_map2(self):
        """Creates the border for a 3D image using 6-connectivity"""
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border


def compute_masd(pred, ref, spacing=None):
    """
    Compute Mean Average Surface Distance between prediction and reference 3D masks.
    Based on MetricsReloaded code.

    Args:
        pred: Binary prediction mask
        ref: Binary reference mask
        spacing: Voxel spacing (optional)
    Returns:
        float: MASD value
    """
    # Input validation
    pred = np.asarray(pred, dtype=np.int8)
    ref = np.asarray(ref, dtype=np.int8)

    if np.sum(pred) == 0 and np.sum(ref) == 0:
        print("Prediction and reference empty - distance set to 0")
        return 0
    elif np.sum(pred) == 0 or np.sum(ref) == 0:
        print("Either prediction or reference empty - MASD undefined")
        return np.nan

    # Get border maps using 3D-specific method
    pred_border = MorphologyOps(pred).border_map2()
    ref_border = MorphologyOps(ref).border_map2()

    # Compute distance transforms
    dt_ref = ndimage.distance_transform_edt(1 - ref_border, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(1 - pred_border, sampling=spacing)

    # Compute mean distances
    mean_pred_to_ref = np.sum(dt_ref[pred_border > 0]) / np.sum(pred_border)
    mean_ref_to_pred = np.sum(dt_pred[ref_border > 0]) / np.sum(ref_border)

    # Final MASD calculation
    masd = 0.5 * (mean_pred_to_ref + mean_ref_to_pred)

    if np.isnan(masd):
        masd = 0.0

    return masd


def compute_standard_metrics(target_mask, pred_mask, spacing=None):
    """Compute MASD along with standard metrics for 3D binary masks"""
    # Convert to binary masks if not already
    target_mask = target_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    # Standard IoU and Dice computation
    intersection = np.logical_and(target_mask, pred_mask).sum()
    union = np.logical_or(target_mask, pred_mask).sum()

    if union == 0:
        return None, None, None

    iou = intersection / union
    dice = 2 * intersection / (target_mask.sum() + pred_mask.sum())

    # Compute MASD
    masd = compute_masd(pred_mask, target_mask, spacing)

    return iou, dice, masd


def compute_scores_and_build_dict(evaluate_on: str, inference_folder: str, fold: int,
                                  trainer_path: str):
    """
    Compute evaluation scores including MASD for 3D medical images.
    """
    try:
        plans_path = join(inference_folder, "plans.pkl")
        if not os.path.exists(plans_path):
            original_path = os.path.normpath(trainer_path)
            splitted_path = original_path.split(os.sep)
            splitted_path[-4] = '_'.join(splitted_path[-4].split('_')[:2])
            first_task_path = '/' + os.path.join(*splitted_path)
            plans_path = join(first_task_path, "plans.pkl")

        plan = load_pickle(plans_path) if os.path.exists(plans_path) else {"num_classes": 2}
    except Exception as e:
        print(f"Error loading plans.pkl: {e}. Using default num_classes=2.")
        plan = {"num_classes": 2}

    num_classes = plan['num_classes']
    dataset_directory = join(preprocessing_output_dir, evaluate_on)
    splits_final = load_pickle(join(dataset_directory, "splits_final.pkl"))
    cases_to_evaluate = splits_final[fold]['test']

    ground_truth_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'labelsTr')
    cases_dict = {}

    for case in cases_to_evaluate:
        file_name = case + ".nii.gz"
        pred_path = join(inference_folder, file_name)
        gt_path = join(ground_truth_folder, file_name)

        assert isfile(pred_path) and isfile(gt_path)

        pred_img = sitk.ReadImage(pred_path)
        gt_img = sitk.ReadImage(gt_path)

        output = sitk.GetArrayFromImage(pred_img)
        target = sitk.GetArrayFromImage(gt_img)
        spacing = pred_img.GetSpacing()[::-1]  # Reverse spacing to match numpy array orientation

        assert np.all(output.shape == target.shape)
        masks_dict = {}

        for c in range(1, num_classes + 1):
            pred_mask = output == c
            target_mask = target == c

            # Compute metrics including MASD with spacing
            iou, dice, masd = compute_standard_metrics(target_mask, pred_mask, spacing)

            masks_dict[f'mask_{c}'] = {
                "IoU": iou,
                "Dice": dice,
                "MASD": masd,
            }

        cases_dict[case] = masks_dict

    return cases_dict


def calculate_aggregated_metrics(val_res: pd.DataFrame):
    """
    Calculate patient-level, task-level, and meta-level aggregate metrics, separated by mask.

    Args:
        val_res: DataFrame containing the validation results

    Returns:
        Tuple of DataFrames (patient_aggregate, task_aggregate, meta_aggregate)
    """
    metrics_to_aggregate = ['IoU', 'Dice', 'MASD']

    # Get unique masks
    unique_masks = val_res['seg_mask'].unique()

    # Patient-level aggregation (across all tasks, per mask)
    patient_aggregate = []
    for mask in unique_masks:
        mask_data = val_res[val_res['seg_mask'] == mask]
        for metric in metrics_to_aggregate:
            metric_data = mask_data[mask_data['metric'] == metric]
            if not metric_data.empty:
                mean = metric_data['value'].mean()
                std = metric_data['value'].std()
                patient_aggregate.append({
                    'Epoch': 'epoch_XXX',
                    'Task': 'ALL_TASKS',
                    'subject_id': 'average',
                    'seg_mask': mask,
                    'metric': metric,
                    'value': mean,
                    'std': std,
                    'aggregation_level': 'patient'
                })

    # Task-level aggregation (per task, per mask)
    task_aggregate = []
    unique_tasks = val_res['Task'].unique()

    for task in unique_tasks:
        task_data = val_res[val_res['Task'] == task]
        for mask in unique_masks:
            mask_data = task_data[task_data['seg_mask'] == mask]
            for metric in metrics_to_aggregate:
                metric_data = mask_data[mask_data['metric'] == metric]
                if not metric_data.empty:
                    mean = metric_data['value'].mean()
                    std = metric_data['value'].std()
                    task_aggregate.append({
                        'Epoch': 'epoch_XXX',
                        'Task': f'{task}_AGGREGATE',
                        'subject_id': 'average',
                        'seg_mask': mask,
                        'metric': metric,
                        'value': mean,
                        'std': std,
                        'aggregation_level': 'task'
                    })

    # Meta-level aggregation (statistics over task-level means and stds, per mask)
    meta_aggregate = []
    task_df = pd.DataFrame(task_aggregate)
    if not task_df.empty:
        for mask in unique_masks:
            mask_data = task_df[task_df['seg_mask'] == mask]
            for metric in metrics_to_aggregate:
                metric_data = mask_data[mask_data['metric'] == metric]
                if not metric_data.empty:
                    # Calculate meta-statistics over means
                    mean_of_means = metric_data['value'].mean()
                    std_of_means = metric_data['value'].std()

                    # Calculate meta-statistics over stds
                    mean_of_stds = metric_data['std'].mean()
                    std_of_stds = metric_data['std'].std()

                    meta_aggregate.append({
                        'Epoch': 'epoch_XXX',
                        'Task': 'META_AGGREGATE',
                        'subject_id': 'average',
                        'seg_mask': mask,
                        'metric': metric,
                        'value': mean_of_means,
                        'std': std_of_means,
                        'mean_of_stds': mean_of_stds,
                        'std_of_stds': std_of_stds,
                        'aggregation_level': 'meta'
                    })

    return pd.DataFrame(patient_aggregate), pd.DataFrame(task_aggregate), pd.DataFrame(meta_aggregate)


def write_summary_report(output_file: str, val_res: pd.DataFrame, network_trainer: str,
                         tasks_joined_name: str, use_head: str, trainer_path: str,
                         fold: int):
    """
    Write a comprehensive summary report including patient, task-level, and meta-level metrics,
    separated by mask/class.
    """
    with open(output_file, 'w') as out:
        # Write header information
        out.write(f'Evaluation performed after Epoch XXX, trained on fold {fold}.\n\n')
        out.write(f"The {network_trainer} model trained on {tasks_joined_name} "
                  f"has been used for this evaluation with the {use_head} head.\n")
        out.write(f"Checkpoint: {join(trainer_path, 'model_final_checkpoint.model')}\n\n")

        # Write task-level aggregates
        out.write("Per-Task Aggregated Metrics:\n")
        out.write("-" * 40 + "\n")
        task_data = val_res[val_res['aggregation_level'] == 'task']
        for task in task_data['Task'].unique():
            out.write(f"\n{task}:\n")
            task_metrics = task_data[task_data['Task'] == task]
            for mask in task_metrics['seg_mask'].unique():
                out.write(f"\nMask {mask}:\n")
                mask_metrics = task_metrics[task_metrics['seg_mask'] == mask]
                for _, row in mask_metrics.iterrows():
                    out.write(f"{row['metric']}:\n")
                    out.write(f"mean ± std: {row['value']:.4f} ± {row['std']:.4f}\n")

        # Write overall patient-level aggregates
        out.write("\nOverall Patient-Level Metrics (across all tasks):\n")
        out.write("-" * 40 + "\n")
        patient_data = val_res[val_res['aggregation_level'] == 'patient']
        for mask in patient_data['seg_mask'].unique():
            out.write(f"\nMask {mask}:\n")
            mask_metrics = patient_data[patient_data['seg_mask'] == mask]
            for _, row in mask_metrics.iterrows():
                out.write(f"{row['metric']}:\n")
                out.write(f"mean ± std: {row['value']:.4f} ± {row['std']:.4f}\n")

        # Write meta-level aggregates
        out.write("\nMeta-Level Statistics (over task-level metrics):\n")
        out.write("-" * 40 + "\n")
        meta_data = val_res[val_res['aggregation_level'] == 'meta']
        for mask in meta_data['seg_mask'].unique():
            out.write(f"\nMask {mask}:\n")
            mask_metrics = meta_data[meta_data['seg_mask'] == mask]
            for _, row in mask_metrics.iterrows():
                out.write(f"{row['metric']}:\n")
                out.write(f"mean of means ± std of means: {row['value']:.4f} ± {row['std']:.4f}\n")
                out.write(f"mean of stds ± std of stds: {row['mean_of_stds']:.4f} ± {row['std_of_stds']:.4f}\n")


def run_evaluation(network: str, network_trainer: str, tasks_list_with_char: Tuple[List[str], str],
                   evaluate_on_tasks: List[str], model_name_joined: str, enable_tta: bool,
                   mixed_precision: bool, chk: Optional[str], fold: int, version: str,
                   vit_type: str, plans_identifier: str, do_LSA: bool, do_SPT: bool,
                   always_use_last_head: bool, use_head: Optional[str], use_model: List[str],
                   extension: str, transfer_heads: bool, use_vit: bool,
                   ViT_task_specific_ln: bool, do_pod: bool,
                   evaluate_initialization: bool, no_delete: bool, legacy_structure: bool):
    """
    Enhanced evaluation function with checkpoint verification and experiment skipping.
    """
    # Fixed inference parameters
    params_ext = {
        'use_head': use_head,
        'always_use_last_head': always_use_last_head,
        'extension': extension,
        'param_split': False,
        'network': network,
        'network_trainer': network_trainer,
        'use_model': use_model,
        'tasks_list_with_char': tasks_list_with_char,
        'plans_identifier': plans_identifier,
        'vit_type': vit_type,
        'version': version
    }

    if evaluate_initialization:
        assert chk is None
        chk = "before_training"
    elif chk is None:
        chk = "model_final_checkpoint"

    tasks_joined_name = join_texts_with_char(tasks_list_with_char[0], tasks_list_with_char[1])
    folder_n = get_ViT_LSA_SPT_folder_name(do_LSA, do_SPT)

    # Find all experiment folders
    experiment_folders = find_experiment_folders(network, network_trainer, tasks_joined_name,
                                                 model_name_joined, plans_identifier)

    for exp_folder in experiment_folders:
        try:
            # Build complete trainer path to check for checkpoint
            if use_vit:
                network_path = join(Generic_ViT_UNet.__name__ + "V" + version, vit_type,
                                    'task_specific' if ViT_task_specific_ln else 'not_task_specific',
                                    folder_n)
            else:
                network_path = Generic_UNet.__name__

            sequence_path = 'SEQ' if transfer_heads else 'MH'
            initial_trainer_path = join(exp_folder, network_path, sequence_path, f"fold_{fold}")
            checkpoint_file = join(initial_trainer_path, f"{chk}.model")

            # Skip this experiment if checkpoint doesn't exist
            if not os.path.exists(checkpoint_file):
                print(f"Skipping experiment {exp_folder} - checkpoint file not found: {checkpoint_file}")
                continue

            output_folders = []
            for evaluate_on in evaluate_on_tasks:
                trainer_path, output_folder = build_trainer_and_output_path(
                    network, network_trainer, exp_folder, tasks_joined_name, model_name_joined,
                    plans_identifier, transfer_heads, folder_n, use_vit,
                    ViT_task_specific_ln, vit_type, version, do_pod,
                    use_head, fold, evaluate_on
                )

                if evaluate_initialization:
                    arr = output_folder.split("/")
                    assert arr[-4] == 'SEQ'
                    assert arr[-5] == Generic_UNet.__name__
                    assert arr[-6] == os.path.basename(exp_folder)
                    assert arr[-7] == model_name_joined
                    arr[-7] = "initialization"
                    output_folder = join("/", *arr)

                input_folder = os.path.join(os.environ['nnUNet_raw_data_base'],
                                            'nnUNet_raw_data', evaluate_on, 'imagesTr')

                # Experiment already was evaluated skip
                final_output_folder = join("/", *output_folder.split('/')[:-1])
                output_file = join(final_output_folder,'summarized_metrics_test.txt')
                if os.path.isfile(output_file):
                    print(f"Skipping experiment {exp_folder} - folder already exists: {output_folder}")
                    continue

                # Create output directory if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)

                try:
                    predict_from_folder(params_ext, trainer_path, input_folder, output_folder,
                                        [fold], False, 1, 2, None, 0, 1, enable_tta,
                                        overwrite_existing=True, mode="normal",
                                        overwrite_all_in_gpu=None, mixed_precision=mixed_precision,
                                        step_size=0.5, checkpoint_name=chk)
                    output_folders.append(output_folder)
                except Exception as e:
                    print(f"Error during prediction for {evaluate_on}:")
                    print(f"Exception type: {type(e).__name__}")
                    print(f"Exception message: {str(e)}")
                    # Clean up the output folder if prediction failed
                    if os.path.exists(output_folder):
                        shutil.rmtree(output_folder)
                    continue

            # Skip metrics calculation if no successful predictions
            if not output_folders:
                print(f"No successful predictions for experiment {exp_folder}, skipping metrics calculation")
                continue

            # Process results for both training and validation data
            file_name = "val_metrics_test"

            if legacy_structure:
                tasks_dict = {}
                for i, evaluate_on in enumerate(evaluate_on_tasks):
                    try:
                        cases_dict = compute_scores_and_build_dict(
                            evaluate_on, output_folders[i], fold,
                            trainer_path=join(exp_folder, "SEQ") if i == 0 else None
                        )
                        tasks_dict[evaluate_on] = cases_dict
                    except Exception as e:
                        print(f"Error computing scores for {evaluate_on}: {str(e)}")
                        continue

                # Skip if no tasks were successfully evaluated
                if not tasks_dict:
                    print(f"No successful evaluations for experiment {exp_folder}")
                    continue

                validation_results = {"epoch_XXX": tasks_dict}
                final_output_folder = join("/", *output_folders[0].split('/')[:-1])

                # Save raw results
                save_json(validation_results, join(final_output_folder, file_name + '.json'),
                          sort_keys=False)

                # Create detailed metrics table
                metrics_data = []
                for epoch, tasks in validation_results.items():
                    for task_name, cases in tasks.items():
                        for case_id, masks in cases.items():
                            for mask_name, metrics in masks.items():
                                # Add standard metrics
                                for metric_name, value in metrics.items():
                                    if metric_name in ['IoU', 'Dice', 'MASD']:
                                        metrics_data.append({
                                            'Epoch': epoch,
                                            'Task': task_name,
                                            'subject_id': case_id,
                                            'seg_mask': mask_name,
                                            'metric': metric_name,
                                            'value': value
                                        })

                val_res = pd.DataFrame(metrics_data)

                # Calculate all levels of aggregates
                patient_agg, task_agg, meta_agg = calculate_aggregated_metrics(val_res)
                val_res = pd.concat([val_res, patient_agg, task_agg, meta_agg], ignore_index=True)

                # Save detailed metrics to CSV
                dumpDataFrameToCsv(val_res, final_output_folder, file_name + '.csv')

                # Generate comprehensive summary report
                output_file = join(final_output_folder,
                                   'summarized_metrics_test.txt')
                write_summary_report(output_file, val_res, network_trainer, tasks_joined_name,
                                     use_head, trainer_path, fold)

            else:  # No legacy structure - separate folders per task
                for i, evaluate_on in enumerate(evaluate_on_tasks):
                    try:
                        tasks_dict = {}
                        cases_dict = compute_scores_and_build_dict(
                            evaluate_on, output_folders[i], fold,
                            trainer_path=join(exp_folder, "SEQ") if i == 0 else None
                        )
                        tasks_dict[evaluate_on] = cases_dict

                        validation_results = {"epoch_XXX": tasks_dict}
                        final_output_folder = join("/", *output_folders[0].split('/')[:-1])
                        task_output_folder = join(final_output_folder, evaluate_on)
                        os.makedirs(task_output_folder, exist_ok=True)

                        # Save raw results
                        save_json(validation_results, join(task_output_folder, file_name + '.json'),
                                  sort_keys=False)

                        # Process metrics for individual task
                        metrics_data = []
                        for epoch, tasks in validation_results.items():
                            for task_name, cases in tasks.items():
                                for case_id, masks in cases.items():
                                    for mask_name, metrics in masks.items():
                                        # Standard metrics
                                        for metric_name, value in metrics.items():
                                            if metric_name in ['IoU', 'Dice', 'MASD']:
                                                metrics_data.append({
                                                    'Epoch': epoch,
                                                    'Task': task_name,
                                                    'subject_id': case_id,
                                                    'seg_mask': mask_name,
                                                    'metric': metric_name,
                                                    'value': value
                                                })

                        val_res = pd.DataFrame(metrics_data)

                        # Calculate all levels of aggregates
                        patient_agg, task_agg, meta_agg = calculate_aggregated_metrics(val_res)
                        val_res = pd.concat([val_res, patient_agg, task_agg, meta_agg], ignore_index=True)

                        # Save detailed metrics to CSV
                        dumpDataFrameToCsv(val_res, task_output_folder, file_name + '.csv')

                        # Generate task-specific summary report
                        output_file = join(task_output_folder,
                                           'summarized_metrics_test.txt')
                        write_summary_report(output_file, val_res, network_trainer, tasks_joined_name,
                                             use_head, trainer_path, fold)

                    except Exception as e:
                        print(f"Error processing task {evaluate_on}: {str(e)}")
                        continue


            if not no_delete:
                for f in output_folders:
                    if os.path.exists(f):
                        shutil.rmtree(f)

        except Exception as e:
            print(f"Error processing experiment {exp_folder}: {str(e)}")
            # Clean up any partial outputs
            for f in output_folders:
                if os.path.exists(f):
                    shutil.rmtree(f)
            continue