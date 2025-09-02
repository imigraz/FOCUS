#########################################################################################################
# ----------This class represents the Evaluation of networks using the extended nnUNet trainer-----------#
# ----------                                     version.                                     -----------#
#########################################################################################################
import glob
import math
import os, argparse, nnunet_ext
import re
from typing import List, Dict, Tuple, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd
from skimage.data import eagle

from nnunet_ext.evaluation.evaluator import Evaluator
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.paths import evaluation_output_dir, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from nnunet_ext.evaluation import evaluator2
from nnunet_ext.evaluation import evaluator3

EXT_MAP = dict()
# -- Extract all extensional trainers in a more generic way -- #
extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if
                  'py' not in x]
for ext in extension_keys:
    trainer_name = \
        [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if
         '.py' in x][0]
    # trainer_keys.extend(trainer_name)
    EXT_MAP[trainer_name] = ext
# -- Add standard trainers as well -- #
EXT_MAP['nnViTUNetTrainer'] = None
EXT_MAP['nnUNetTrainerV2'] = 'standard'
EXT_MAP['nnViTUNetTrainerCascadeFullRes'] = None


def run_evaluation(evaluator: str):
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert evaluation_output_dir is not None, "Before running any evaluation, please specify the Evaluation folder (EVALUATION_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf

    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)

    # -- Additional arguments specific for multi head training -- #
    parser.add_argument("-f", "--folds", action='store', type=str, nargs="+",
                        help="Specify on which folds to train on. Use a fold between 0, 1, ..., 4 or \'all\'",
                        required=True)
    parser.add_argument("-trained_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-use_model", "--use", action='store', type=str, nargs="+",
                        help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation of tasks the network is not trained on. When using a non nn-UNet extension, that' +
                             'is not necessary. If this is not set, always the latest trained head will be used.')
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify if mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network will be evaluated on. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. ' +
                             ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.' +
                             ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('--store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                             ' as a .csv file as well. Default: .csv are not created.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are only four' +
                             ' possibilities: 1, 2, 3 or 4.' +
                             ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base',
                        choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three' +
                             ' possibilities: base, large or huge.' +
                             ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. ' +
                             'Note that then the flags -v, -v_type and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')
    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                             ' during training, ie. the very first head from the initialization of the class is used.'
                             ' Default: The previously trained head is used as initialization of the new head.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. ' +
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument('--always_use_last_head', action='store_true', default=False,
                        help='If this is set, during the evaluation, always the last head will be used, ' +
                             'for every dataset the evaluation is performed on. When an extension network was trained with ' +
                             'the -transfer_heads flag then this should be set, i.e. nnUNetTrainerSequential or nnUNetTrainerFrozendViT.')
    parser.add_argument('--no_pod', action='store_true', default=False,
                        help='This will only be considered if our own trainers are used. If set, this flag indicates that the POD ' +
                             'embedding has not been used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--adaptive', required=False, default=False, action="store_true",
                        help='Set this flag if the EWC loss has been changed during the frozen training process (ewc_lambda*e^{-1/3}). '
                             ' Default: The EWC loss will not be altered. --> Makes only sense with our nnUNetTrainerFrozEWC trainer.')
    parser.add_argument('--include_training_data', action='store_true', default=False,
                        help='Set this flag if the evaluation should also be done on the training data.')

    parser.add_argument("--enable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument('-chk',
                        help='checkpoint name, model_final_checkpoint' or 'model_best',
                        required=False,
                        default=None)
    parser.add_argument('-evaluate_initialization', required=False, default=False, action="store_true",
                        help="set this evaluate the random initialization")
    parser.add_argument('-no_delete', required=False, default=False, action="store_true",
                        help="set this to not delete the inference files")
    parser.add_argument('-legacy_structure', required=False, default=False, action="store_true", help="set this ")

    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    always_use_last_head = args.always_use_last_head

    # -- Extract the arguments specific for all trainers from argument parser -- #
    trained_on = args.trained_on  # List of the tasks that helps to navigate to the correct folder, eg. A B C
    use_model = args.use  # List of the tasks representing the network to use, e. use A B from folder A B C
    evaluate_on = args.evaluate_on  # List of the tasks that should be used to evaluate the model
    use_head = args.use_head  # One task specifying which head should be used
    if isinstance(use_head, list):
        use_head = use_head[0]

    # -- Extract further arguments -- #
    save_csv = args.store_csv
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used
    transfer_heads = not args.no_transfer_heads
    do_pod = not args.no_pod
    adaptive = args.adaptive

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln

    # -- Extract the vit_type structure and check it is one from the existing ones -- #s
    vit_type = args.vit_type
    if isinstance(vit_type,
                  list):  # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT

    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error

    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'

    cuda = join_texts_with_char(cuda, ',')

    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Set bool if user wants to use train data during eval as well -- #
    use_all_data = args.include_training_data

    # -- Extract the desired version, only considered in case of ViT Trainer -- #
    version = args.version
    if isinstance(version,
                  list):  # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]

    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(5))
    else:  # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int,
                             list)), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(
        network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    use_model_w_tasks = list()
    evaluate_on_tasks = list()
    if use_head is not None:
        use_head = convert_id_to_task_name(int(use_head)) if not use_head.startswith("Task") else use_head
    for idx, t in enumerate(trained_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        tasks_for_folder.append(t)
    for idx, t in enumerate(use_model):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        use_model_w_tasks.append(t)
    for idx, t in enumerate(evaluate_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        evaluate_on_tasks.append(t)

    char_to_join_tasks = '_'

    # ---------------------------------------------
    # Evaluate for each task and all provided folds
    # ---------------------------------------------
    if evaluator == 'Evaluator':
        evaluator = Evaluator(network, network_trainer, (tasks_for_folder, char_to_join_tasks),
                              (use_model_w_tasks, char_to_join_tasks),
                              version, vit_type, plans_identifier, mixed_precision, EXT_MAP[network_trainer], save_csv,
                              transfer_heads,
                              use_vit, False, ViT_task_specific_ln, do_LSA, do_SPT)
        evaluator.evaluate_on(fold, evaluate_on_tasks, use_head, always_use_last_head, do_pod=do_pod, adaptive=adaptive,
                              use_all_data=use_all_data)
    elif evaluator == 'evaluator2':
        model_name_joined = join_texts_with_char(use_model_w_tasks, char_to_join_tasks)

        if args.evaluate_initialization:
            assert len(use_model_w_tasks) == 1, "need to speficy first task to do evaluate the random init"

        for f in fold:
            evaluator2.run_evaluation2(network, network_trainer, (tasks_for_folder, char_to_join_tasks),
                                       evaluate_on_tasks, model_name_joined, args.enable_tta, mixed_precision, args.chk,
                                       f,
                                       version, vit_type, plans_identifier, do_LSA, do_SPT, always_use_last_head,
                                       use_head, use_model, EXT_MAP[network_trainer], transfer_heads,
                                       use_vit, ViT_task_specific_ln, do_pod, args.include_training_data,
                                       args.evaluate_initialization, args.no_delete, args.legacy_structure)
    else:
        model_name_joined = join_texts_with_char(use_model_w_tasks, char_to_join_tasks)

        if args.evaluate_initialization:
            assert len(use_model_w_tasks) == 1, "need to speficy first task to do evaluate the random init"

        for f in fold:
            evaluator3.run_evaluation3(network, network_trainer, (tasks_for_folder, char_to_join_tasks),
                                       evaluate_on_tasks, model_name_joined, args.enable_tta, mixed_precision, args.chk,
                                       f,
                                       version, vit_type, plans_identifier, do_LSA, do_SPT, always_use_last_head,
                                       use_head, use_model, EXT_MAP[network_trainer], transfer_heads,
                                       use_vit, ViT_task_specific_ln, do_pod, args.include_training_data,
                                       args.evaluate_initialization, args.no_delete, args.legacy_structure)


# -- Main function for setup execution -- #
def main():
    run_evaluation('Evaluator')


def run_evaluation2():
    run_evaluation('evaluator2')


def run_evaluation3():
    run_evaluation('evaluator3')


def run_evaluation_all():
    trainers = ["nnUNetTrainerFOCUS", "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "nnUNetTrainerSequentialGIN",
                "nnUNetTrainerSequential", "nnUNetTrainerTED", "nnUNetTrainerEWC", "nnUNetTrainerMiB"]

    task_ids_all = [[111]]
    task_ids_all.append([112])
    task_ids_all.append([113])
    task_ids_all.append([115])
    task_ids_all.append([197])
    task_ids_all.append([198])
    task_ids_all.append([199])
    task_ids_all.append([111, 112, 113, 115])
    task_ids_all.append([197, 198])
    task_ids_all.append([197, 198, 199])
    for task_ids in task_ids_all:
        for i in list(range(1)):
            eval_task(trainers, task_ids, fold=i)

        eval_task_crossvalidation(trainers, task_ids)


def extract_metrics_from_csv_multi_mask(file_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract DSC and MASD metrics from a validation metrics CSV file for all masks.

    Args:
        file_path (str): Path to the metrics CSV file

    Returns:
        Dict: Dictionary with task names as keys, mask names as secondary keys, and their metrics as values
        Structure: {task_name: {mask_name: {'dice': value, 'masd': value}}}
    """
    metrics = {}
    print(f"\nProcessing file: {file_path}")

    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='\t')
        print(f"Columns in CSV: {df.columns.tolist()}")
        print(f"Unique Tasks in CSV: {df['Task'].unique()}")

        # Filter for task-level aggregates and relevant metrics
        task_aggregates = df[
            (df['Task'].str.contains('_AGGREGATE', na=False)) &
            (df['subject_id'] == 'average')
            ]
        print(f"Found {len(task_aggregates)} task aggregate rows")

        # Process each task
        for task_full in task_aggregates['Task'].unique():
            print(f"Processing task: {task_full}")
            base_task = task_full.replace('_AGGREGATE', '')
            task_rows = task_aggregates[task_aggregates['Task'] == task_full]

            dice_rows = task_rows[task_rows['metric'] == 'Dice']
            masd_rows = task_rows[task_rows['metric'] == 'MASD']

            if not dice_rows.empty and not masd_rows.empty:
                metrics[base_task] = {}

                # Process all masks (each row represents a different mask)
                for idx, (dice_row, masd_row) in enumerate(zip(dice_rows.itertuples(), masd_rows.itertuples())):
                    mask_name = f"mask_{idx + 1}"  # or extract from a column if available

                    masd_score = float(metrics[base_task][mask_name]['masd'])
                    if np.isnan(masd_score):
                        masd_score = 0.0
                    metrics[base_task][mask_name] = {
                        'dice': float(dice_row.value),
                        'masd': masd_score,
                    }
                    print(
                        f"Extracted metrics for {base_task} {mask_name}: "
                        f"Dice={metrics[base_task][mask_name]['dice']:.4f}, "
                        f"MASD={masd_score:.4f}"
                    )

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}

    return metrics


def compute_transfer_metrics_multi_mask(exp_folders: Dict[str, Dict[str, List]],
                                        seq_folders: Dict[str, Dict[str, List]],
                                        tasks: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute BWT, FWT, and AVG metrics across all time steps and domains for multiple masks.

    Returns:
        Dict: Structure {experiment_name: {mask_name: {metric_name: value}}}
    """
    transfer_metrics = {}
    print(f"\nTasks to process: {tasks}")

    # Get single task models from sequential trainer for FWT computation
    single_task_metrics = {}
    for task in tasks[1:]:  # Skip first task as we don't need it for FWT
        search_key = task
        if 'seq' in seq_folders and search_key in seq_folders['seq']:
            metrics_path = join(seq_folders['seq'][search_key][1], "val_metrics_test.csv")
            if os.path.exists(metrics_path):
                print(f"Found single task model for task {task} in sequential trainer folder")
                single_task_metrics[task] = extract_metrics_from_csv_multi_mask(metrics_path)
            else:
                print(f"No metrics file found for single task {task} in sequential trainer folder")
        else:
            print(f"No single task model found for task {task} in sequential trainer folder")

    # Process each experiment except 'seq'
    for exp_name, task_paths in exp_folders.items():
        print(f"\nProcessing experiment: {exp_name}")
        print(f"Available task paths: {[k for k in task_paths.keys()]}")

        transfer_metrics[exp_name] = {}

        # First pass: determine all available masks by checking the first available metrics file
        available_masks = set()
        for task_sequence_key in task_paths.keys():
            metrics_path = join(task_paths[task_sequence_key][1], "val_metrics_test.csv")
            if os.path.exists(metrics_path):
                temp_metrics = extract_metrics_from_csv_multi_mask(metrics_path)
                for task_metrics in temp_metrics.values():
                    available_masks.update(task_metrics.keys())
                break

        print(f"Available masks: {available_masks}")

        # Initialize metrics structure for each mask
        for mask_name in available_masks:
            transfer_metrics[exp_name][mask_name] = {
                'bwt_measurements': [],  # Store all BWT measurements with their context
                'fwt_dice': [],
                'fwt_masd': [],
                'avg_timestep_dice': [],  # Store average dice for each timestep
                'avg_timestep_masd': [],  # Store average masd for each timestep
                'avg_all_dice': [],  # Store all individual dice scores for BWT-style averaging
                'avg_all_masd': [],  # Store all individual masd scores for BWT-style averaging
            }

        # Process each task sequence length
        for i in range(len(tasks)):
            current_task = tasks[i]
            current_sequence = tasks[:i + 1]
            current_sequence_key = '_'.join(current_sequence)

            if current_sequence_key not in task_paths:
                print(f"Sequence {current_sequence_key} not found in paths")
                continue

            current_metrics_path = join(task_paths[current_sequence_key][1], "val_metrics_test.csv")
            if not os.path.exists(current_metrics_path):
                print(f"Metrics file not found at {current_metrics_path}")
                continue

            print(f"\nProcessing sequence up to task {current_task}")
            current_metrics = extract_metrics_from_csv_multi_mask(current_metrics_path)

            # Compute AVG metrics for this timestep
            for mask_name in available_masks:
                # Collect dice and masd scores for all tasks seen so far at this timestep
                timestep_dice_scores = []
                timestep_masd_scores = []

                for seen_task in current_sequence:
                    if seen_task in current_metrics and mask_name in current_metrics[seen_task]:
                        dice_score = current_metrics[seen_task][mask_name]['dice']
                        masd_score = current_metrics[seen_task][mask_name]['masd']

                        if np.isnan(masd_score):
                            masd_score = 0.0

                        timestep_dice_scores.append(dice_score)
                        timestep_masd_scores.append(masd_score)

                        # Also add to the list for BWT-style averaging
                        transfer_metrics[exp_name][mask_name]['avg_all_dice'].append(dice_score)
                        transfer_metrics[exp_name][mask_name]['avg_all_masd'].append(masd_score)

                # Compute average for this timestep
                if timestep_dice_scores:
                    avg_dice_timestep = np.mean(timestep_dice_scores)
                    avg_masd_timestep = np.mean(timestep_masd_scores)

                    transfer_metrics[exp_name][mask_name]['avg_timestep_dice'].append(avg_dice_timestep)
                    transfer_metrics[exp_name][mask_name]['avg_timestep_masd'].append(avg_masd_timestep)

                    print(
                        f"Timestep {i + 1} AVG for {mask_name}: Dice={avg_dice_timestep:.4f}, MASD={avg_masd_timestep:.4f}")

            # BWT computation
            if i > 0:  # Skip first task as it has no BWT
                for prev_idx, prev_task in enumerate(tasks[:i]):
                    prev_sequence = tasks[:prev_idx + 1]
                    prev_sequence_key = '_'.join(prev_sequence)

                    if prev_sequence_key not in task_paths:
                        print(f"Previous sequence {prev_sequence_key} not found in paths")
                        continue

                    prev_metrics_path = join(task_paths[prev_sequence_key][1], "val_metrics_test.csv")
                    if not os.path.exists(prev_metrics_path):
                        print(f"Previous metrics file not found at {prev_metrics_path}")
                        continue

                    print(f"Computing BWT for {prev_task} after learning {current_task}")
                    original_metrics = extract_metrics_from_csv_multi_mask(prev_metrics_path)

                    # Process each mask separately
                    for mask_name in available_masks:
                        if (prev_task in original_metrics and mask_name in original_metrics[prev_task] and
                                prev_task in current_metrics and mask_name in current_metrics[prev_task]):
                            bwt_measurement = {
                                'prev_task': prev_task,
                                'current_task': current_task,
                                'time_step': i,
                                'dice_diff': current_metrics[prev_task][mask_name]['dice'] -
                                             original_metrics[prev_task][mask_name]['dice'],
                                'masd_diff': current_metrics[prev_task][mask_name]['masd'] -
                                             original_metrics[prev_task][mask_name]['masd']
                            }
                            transfer_metrics[exp_name][mask_name]['bwt_measurements'].append(bwt_measurement)
                            print(f"BWT measurement recorded for {mask_name}: Task {prev_task} after {current_task}: " +
                                  f"Dice={bwt_measurement['dice_diff']:.4f}, MASD={bwt_measurement['masd_diff']:.4f}")

                # FWT computation
                for mask_name in available_masks:
                    if (current_task in single_task_metrics and
                            current_task in single_task_metrics[current_task] and
                            mask_name in single_task_metrics[current_task][current_task] and
                            current_task in current_metrics and
                            mask_name in current_metrics[current_task]):

                        print(f"Computing FWT for {current_task} {mask_name} using sequential trainer reference")
                        fwt_dice = (current_metrics[current_task][mask_name]['dice'] -
                                    single_task_metrics[current_task][current_task][mask_name]['dice'])
                        fwt_masd = (current_metrics[current_task][mask_name]['masd'] -
                                    single_task_metrics[current_task][current_task][mask_name]['masd'])

                        transfer_metrics[exp_name][mask_name]['fwt_dice'].append(fwt_dice)
                        transfer_metrics[exp_name][mask_name]['fwt_masd'].append(fwt_masd)
                        print(f"FWT for {current_task} {mask_name}: Dice={fwt_dice:.4f}, MASD={fwt_masd:.4f}")
                    else:
                        print(f"Skipping FWT for {current_task} {mask_name}: No single task reference model available")

        # Calculate final averages for each mask
        for mask_name in available_masks:
            mask_metrics = transfer_metrics[exp_name][mask_name]

            # Calculate BWT averages
            all_dice_diffs = [m['dice_diff'] for m in mask_metrics['bwt_measurements']]
            all_masd_diffs = [m['masd_diff'] for m in mask_metrics['bwt_measurements']]

            if all_dice_diffs:
                mask_metrics['bwt_dice_mean'] = np.mean(all_dice_diffs)
                mask_metrics['bwt_dice_std'] = np.std(all_dice_diffs)
                mask_metrics['bwt_masd_mean'] = np.mean(all_masd_diffs)
                mask_metrics['bwt_masd_std'] = np.std(all_masd_diffs)
                print(f"\nOverall BWT metrics for {mask_name}:")
                print(f"Dice - Mean ± Std: {mask_metrics['bwt_dice_mean']:.4f} ± {mask_metrics['bwt_dice_std']:.4f}")
                print(f"MASD - Mean ± Std: {mask_metrics['bwt_masd_mean']:.4f} ± {mask_metrics['bwt_masd_std']:.4f}")
            else:
                print(f"\nNo BWT measurements available for {mask_name}")
                mask_metrics['bwt_dice_mean'] = 0.0
                mask_metrics['bwt_dice_std'] = 0.0
                mask_metrics['bwt_masd_mean'] = 0.0
                mask_metrics['bwt_masd_std'] = 0.0

            # Calculate AVG metrics - Equal timestep weighting
            if mask_metrics['avg_timestep_dice']:
                mask_metrics['avg_dice_timestep_equal'] = np.mean(mask_metrics['avg_timestep_dice'])
                mask_metrics['avg_dice_timestep_equal_std'] = np.std(mask_metrics['avg_timestep_dice'])
                mask_metrics['avg_masd_timestep_equal'] = np.mean(mask_metrics['avg_timestep_masd'])
                mask_metrics['avg_masd_timestep_equal_std'] = np.std(mask_metrics['avg_timestep_masd'])
                print(f"\nAVG metrics (equal timestep weighting) for {mask_name}:")
                print(
                    f"Dice: {mask_metrics['avg_dice_timestep_equal']:.4f} ± {mask_metrics['avg_dice_timestep_equal_std']:.4f}")
                print(
                    f"MASD: {mask_metrics['avg_masd_timestep_equal']:.4f} ± {mask_metrics['avg_masd_timestep_equal_std']:.4f}")
            else:
                mask_metrics['avg_dice_timestep_equal'] = 0.0
                mask_metrics['avg_dice_timestep_equal_std'] = 0.0
                mask_metrics['avg_masd_timestep_equal'] = 0.0
                mask_metrics['avg_masd_timestep_equal_std'] = 0.0

            # Calculate AVG metrics - BWT-style weighting (all individual scores)
            if mask_metrics['avg_all_dice']:
                mask_metrics['avg_dice_bwt_style'] = np.mean(mask_metrics['avg_all_dice'])
                mask_metrics['avg_dice_bwt_style_std'] = np.std(mask_metrics['avg_all_dice'])
                mask_metrics['avg_masd_bwt_style'] = np.mean(mask_metrics['avg_all_masd'])
                mask_metrics['avg_masd_bwt_style_std'] = np.std(mask_metrics['avg_all_masd'])
                print(f"\nAVG metrics (BWT-style weighting) for {mask_name}:")
                print(f"Dice: {mask_metrics['avg_dice_bwt_style']:.4f} ± {mask_metrics['avg_dice_bwt_style_std']:.4f}")
                print(f"MASD: {mask_metrics['avg_masd_bwt_style']:.4f} ± {mask_metrics['avg_masd_bwt_style_std']:.4f}")
            else:
                mask_metrics['avg_dice_bwt_style'] = 0.0
                mask_metrics['avg_dice_bwt_style_std'] = 0.0
                mask_metrics['avg_masd_bwt_style'] = 0.0
                mask_metrics['avg_masd_bwt_style_std'] = 0.0

            # Calculate FWT averages (existing code)
            for metric in ['fwt_dice', 'fwt_masd']:
                values = mask_metrics[metric]
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    mask_metrics[f'{metric}_mean'] = mean_value
                    mask_metrics[f'{metric}_std'] = std_value
                    print(f"\n{mask_name} {metric} summary - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
                else:
                    mask_metrics[f'{metric}_mean'] = 0.0
                    mask_metrics[f'{metric}_std'] = 0.0

    return transfer_metrics


def append_transfer_metrics_to_summary_multi_mask(output_file: str,
                                                  transfer_metrics: Dict[str, Dict[str, Dict[str, float]]],
                                                  tasks: List[str]):
    """
    Append transfer metrics to the existing summary file with temporal BWT measurements, AVG metrics,
    and detailed debugging information for multiple masks.
    """
    with open(output_file, 'a') as f:
        f.write("\nTransfer Learning Metrics (Multi-Mask):\n")
        f.write("-" * 80 + "\n")

        for exp_name, mask_metrics in transfer_metrics.items():
            f.write(f"\nExperiment: {exp_name}\n")

            for mask_name, metrics in mask_metrics.items():
                f.write(f"\n--- {mask_name.upper()} ---\n")

                # Write AVG metrics summary
                f.write("\nAverage Performance (AVG) - Summary:\n")
                f.write(
                    f"Equal timestep weighting - Dice: {metrics['avg_dice_timestep_equal']:.4f} ± {metrics['avg_dice_timestep_equal_std']:.4f}, MASD: {metrics['avg_masd_timestep_equal']:.4f} ± {metrics['avg_masd_timestep_equal_std']:.4f}\n")
                f.write(
                    f"BWT-style weighting - Dice: {metrics['avg_dice_bwt_style']:.4f} ± {metrics['avg_dice_bwt_style_std']:.4f}, MASD: {metrics['avg_masd_bwt_style']:.4f} ± {metrics['avg_masd_bwt_style_std']:.4f}\n")

                # Write detailed AVG computation debugging info
                f.write("\nAVG Computation Details (for debugging):\n")
                f.write("-" * 40 + "\n")

                # Equal timestep weighting details
                if 'avg_timestep_dice' in metrics and metrics['avg_timestep_dice']:
                    f.write("Equal timestep weighting breakdown:\n")
                    f.write("Timestep averages used for final AVG calculation:\n")
                    for i, (dice_avg, masd_avg) in enumerate(
                            zip(metrics['avg_timestep_dice'], metrics['avg_timestep_masd'])):
                        f.write(f"  Timestep {i + 1} (after {tasks[i]}): Dice={dice_avg:.4f}, MASD={masd_avg:.4f}\n")

                    f.write(
                        f"Final calculation: AVG_dice = ({' + '.join([f'{x:.4f}' for x in metrics['avg_timestep_dice']])}) / {len(metrics['avg_timestep_dice'])} = {metrics['avg_dice_timestep_equal']:.4f}\n")
                    f.write(
                        f"Final calculation: AVG_masd = ({' + '.join([f'{x:.4f}' for x in metrics['avg_timestep_masd']])}) / {len(metrics['avg_timestep_masd'])} = {metrics['avg_masd_timestep_equal']:.4f}\n")

                # BWT-style weighting details
                if 'avg_all_dice' in metrics and metrics['avg_all_dice']:
                    f.write(f"\nBWT-style weighting breakdown:\n")
                    f.write(f"All individual domain performances used (n={len(metrics['avg_all_dice'])}):\n")

                    # Group by timestep for clearer presentation
                    dice_idx = 0
                    masd_idx = 0
                    for i in range(len(tasks)):
                        current_sequence = tasks[:i + 1]
                        f.write(f"  Timestep {i + 1} (after {tasks[i]}) - {len(current_sequence)} values:\n")
                        timestep_dice = []
                        timestep_masd = []
                        for j in range(len(current_sequence)):
                            if dice_idx < len(metrics['avg_all_dice']):
                                dice_val = metrics['avg_all_dice'][dice_idx]
                                masd_val = metrics['avg_all_masd'][masd_idx]
                                f.write(f"    Task {current_sequence[j]}: Dice={dice_val:.4f}, MASD={masd_val:.4f}\n")
                                timestep_dice.append(dice_val)
                                timestep_masd.append(masd_val)
                                dice_idx += 1
                                masd_idx += 1

                    f.write(
                        f"Final calculation: AVG_dice = sum({len(metrics['avg_all_dice'])} values) / {len(metrics['avg_all_dice'])} = {metrics['avg_dice_bwt_style']:.4f}\n")
                    f.write(
                        f"Final calculation: AVG_masd = sum({len(metrics['avg_all_masd'])} values) / {len(metrics['avg_all_masd'])} = {metrics['avg_masd_bwt_style']:.4f}\n")

                f.write("-" * 40 + "\n")

                # Write average BWT metrics (across all measurements)
                f.write("\nBackward Transfer (BWT) - Average across all measurements:\n")
                f.write(f"Dice - Mean ± Std: {metrics['bwt_dice_mean']:.4f} ± {metrics['bwt_dice_std']:.4f}\n")
                f.write(f"MASD - Mean ± Std: {metrics['bwt_masd_mean']:.4f} ± {metrics['bwt_masd_std']:.4f}\n")

                # Write FWT metrics if available
                if metrics.get('fwt_dice'):
                    f.write("\nForward Transfer (FWT):\n")
                    f.write(f"Dice - Mean ± Std: {metrics['fwt_dice_mean']:.4f} ± {metrics['fwt_dice_std']:.4f}\n")
                    f.write(f"MASD - Mean ± Std: {metrics['fwt_masd_mean']:.4f} ± {metrics['fwt_masd_std']:.4f}\n")

                # Write detailed BWT measurements chronologically
                f.write("\nDetailed BWT Measurements (chronological order):\n")
                # Sort measurements by time step and task order
                sorted_measurements = sorted(metrics['bwt_measurements'],
                                             key=lambda x: (x['time_step'], tasks.index(x['prev_task'])))

                current_time_step = None
                for measurement in sorted_measurements:
                    # Add header for new time step
                    if current_time_step != measurement['time_step']:
                        current_time_step = measurement['time_step']
                        f.write(
                            f"\nAfter learning task {measurement['current_task']} (time step {current_time_step + 1}):\n")

                    f.write(f"  Task {measurement['prev_task']}: " +
                            f"Dice diff={measurement['dice_diff']:.4f}, " +
                            f"MASD diff={measurement['masd_diff']:.4f}\n")

                # Write individual FWT measurements if available
                if metrics.get('fwt_dice'):
                    f.write("\nIndividual FWT Measurements:\n")
                    for task, dice, masd in zip(tasks[1:], metrics['fwt_dice'], metrics['fwt_masd']):
                        f.write(f"Task {task} (continual vs. single-task): " +
                                f"Dice={dice:.4f}, MASD={masd:.4f}\n")


def compute_btw_for_trainer_multi_mask(trainer_name: str, tasks: List[str], evaluation_folder: str,
                                       fold: int = 0, res: str = "2d"):
    """
    Compute BTW metrics for all experiments of a specific trainer with multi-mask support and append to summary files.

    Args:
        trainer_name (str): Name of the trainer
        tasks (List[str]): List of tasks in training order
        evaluation_folder (str): Base evaluation folder path
        fold (int): Fold number to analyze
        res (str): Resolution type
    """
    print(f"\nProcessing BTW metrics for trainer: {trainer_name}")

    # Find all experiment folders for this trainer
    exp_folders = find_experiment_folders(evaluation_folder, trainer_name, tasks, res, fold)

    # First, find sequential trainer folders for single task models
    seq_folders = {}
    for task in tasks[1:]:
        dic = find_experiment_folders(evaluation_folder, "nnUNetTrainerSequential", [task], res, fold)
        if 'seq' in dic:
            if len(seq_folders) == 0:
                seq_folders['seq'] = dic['seq']
            else:
                seq_folders['seq'].update(dic['seq'])

    if not exp_folders:
        print(f"No experiment folders found for trainer {trainer_name}")
        return

    # Process transfer metrics for each experiment with multi-mask support
    transfer_metrics = compute_transfer_metrics_multi_mask(exp_folders, seq_folders, tasks)

    # Append transfer metrics to each summary file
    for exp_name, task_paths in exp_folders.items():
        for task_path in task_paths.values():
            summary_file = join(task_path[1], "summarized_metrics_test.txt")
            if os.path.exists(summary_file):
                append_transfer_metrics_to_summary_multi_mask(summary_file, {exp_name: transfer_metrics[exp_name]},
                                                              tasks)

    return transfer_metrics

def generate_evaluation_scenarios(tasks):
    """
    Generates all possible evaluation scenarios for continual learning based on task sequence.
    Each scenario represents a checkpoint in the continual learning process.

    Args:
        tasks (list): List of task names/ids in learning sequence order

    Returns:
        list: List of scenario dictionaries containing evaluation parameters
    """
    scenarios = []

    # Generate scenarios for each step in the continual learning sequence
    for i in range(len(tasks)):
        task_subset = tasks[:i + 1]
        scenarios.append({
            'use_model': task_subset,
            'evaluate_initialization': False,
            'chk': None,
            'description': f'Evaluation after training on {len(task_subset)} task(s): {", ".join(task_subset)}'
        })

    return scenarios


def eval_task(trainers, task_ids, gpu: str = "0", res: str = "2d", fold: int = 0):
    """
    Run evaluation for datasets using merged evaluator with specific trainer.
    Automatically generates and evaluates all task combinations for continual learning.

    Args:
        trainers (list): List of trainer names to evaluate
        task_ids (list): List of task IDs in sequence order
        gpu (str): GPU device ID
        res (str): Resolution, default "2d"
        fold (int): which nnUNet fold to evaluate 0-4
    """
    # Set environment variables
    evaluation_folder = os.environ["EVALUATION_FOLDER"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(os.environ["EVALUATION_FOLDER"], exist_ok=True)

    for trainer in trainers:
        # Convert task IDs to task names
        tasks = [convert_id_to_task_name(task_id) for task_id in task_ids]
        tasks_with_char = (tasks, '_')
        evaluate_on = tasks.copy()

        # Generate all evaluation scenarios
        scenarios = generate_evaluation_scenarios(tasks)

        use_vit = False
        if 'ViT' in trainer:
            use_vit = True

        # Run evaluation for each scenario
        for scenario in scenarios:
            print(f"\nRunning evaluation for scenario: {scenario['description']}")
            print(f"Models being evaluated: {scenario['use_model']}")

            evaluator3.run_evaluation(
                   network=res,
                   network_trainer=trainer,
                   tasks_list_with_char=tasks_with_char,
                   evaluate_on_tasks=evaluate_on,
                   model_name_joined='_'.join(scenario['use_model']),
                   enable_tta=False,
                   mixed_precision=True,
                   chk=scenario.get('chk'),
                   fold=fold,
                   version='1',
                   vit_type='base',
                   plans_identifier='nnUNetPlansv2.1',
                   do_LSA=False,
                   do_SPT=False,
                   always_use_last_head=False,
                   use_head=None,
                   use_model=scenario['use_model'],
                   extension=EXT_MAP[trainer],
                   transfer_heads=True,
                   use_vit=use_vit,
                   ViT_task_specific_ln=False,
                   do_pod=False,
                   evaluate_initialization=scenario.get('evaluate_initialization', False),
                   no_delete=True,
                   legacy_structure=True)

        if len(tasks) > 1:
            # Compute BTW metrics for all experiments of this trainer
            compute_btw_for_trainer_multi_mask(trainer, tasks, evaluation_folder, fold=fold, res=res)# Pass resolution type


def find_experiment_folders(base_path: str, trainer_name: str, tasks: List[str], res: str = "2d", fold = 0) -> Dict[
    str, Dict[str, List]]:
    """
    Find and group experiment folders by experiment name/timestamp, including single task models.

    Args:
        base_path (str): Base evaluation path
        trainer_name (str): Name of the trainer
        tasks (List[str]): List of tasks in training order
        res (str): Resolution type (e.g., "2d", "3d")

    Returns:
        Dict[str, Dict[str, List]]: Dictionary mapping experiment names to their folders for each task combination
    """
    # Dictionary to store all folders grouped by experiment name
    experiments_dict = {}

    # Find training folders
    all_tasks_path = '_'.join(tasks)
    for i in range(len(tasks)):
        current_tasks = tasks[:(i + 1)]
        current_tasks_path = '_'.join(current_tasks)

        search_pattern = join(base_path, 'nnUNet_ext', res, all_tasks_path, current_tasks_path,
                              f"{trainer_name}__nnUNetPlansv2.1*")

        matching_folders = glob.glob(search_pattern)

        for folder in matching_folders:
            folder_name = os.path.basename(folder)
            exp_name = folder_name.split('__nnUNetPlansv2.1_')[1]
            parts = exp_name.split('_')
            timestamp = '_'.join(parts[0:2])
            date_object = datetime.strptime(timestamp, "%d.%m.%Y_%H:%M:%S")

            if len(parts) < 3: # no experiment name, we skip
                continue
            key = '_'.join(parts[2:])  # Use the actual experiment name for sequential models
            if len(parts) == 3:
                key = parts[2]
            task_dic = {}
            folder = join(folder, "Generic_UNet", "SEQ", "head_None", f"fold_{fold}")
            if 'ViT' in trainer_name:
                folder = join(folder, "Generic_ViT_UNetV1", "SEQ", "head_None", f"fold_{fold}")
            entry = [date_object, folder]
            task_dic[current_tasks_path] = entry

            if key not in experiments_dict:
                experiments_dict[key] = task_dic
            elif current_tasks_path not in experiments_dict[key]:
                experiments_dict[key][current_tasks_path] = entry
            elif experiments_dict[key][current_tasks_path][0] < entry[0]:
                experiments_dict[key][current_tasks_path] = entry

    # Print summary of found experiments
    print("\nFound experiments:")
    for exp_name, task_dict in experiments_dict.items():
        print(f"\nExperiment: {exp_name}")
        print(f"Tasks: {list(task_dict.keys())}")

    return experiments_dict


def find_experiment_folders_with_folds(base_path: str, trainer_name: str, tasks: List[str], res: str = "2d") -> Dict[
    str, Dict[str, Dict[int, List]]]:
    """
    Find and group experiment folders by experiment name/timestamp, collecting all folds for each task combination.

    Returns:
        Dict[str, Dict[str, Dict[int, List]]]: Dictionary mapping experiment names to task combinations to fold dictionaries
        Structure: {exp_name: {task_combination: {fold_num: [date_object, folder_path]}}}
    """
    experiments_dict = {}

    all_tasks_path = '_'.join(tasks)
    for i in range(len(tasks)):
        current_tasks = tasks[:(i + 1)]
        current_tasks_path = '_'.join(current_tasks)

        search_pattern = join(base_path, 'nnUNet_ext', res, all_tasks_path, current_tasks_path,
                              f"{trainer_name}__nnUNetPlansv2.1*")

        matching_folders = glob.glob(search_pattern)

        for folder in matching_folders:
            folder_name = os.path.basename(folder)
            exp_name = folder_name.split('__nnUNetPlansv2.1_')[1]
            parts = exp_name.split('_')
            timestamp = '_'.join(parts[0:2])
            date_object = datetime.strptime(timestamp, "%d.%m.%Y_%H:%M:%S")

            if len(parts) < 3:
                continue

            key = '_'.join(parts[2:])
            if len(parts) == 3:
                key = parts[2]

            # Find all fold folders for this experiment
            base_folder = join(folder, "Generic_UNet", "SEQ", "head_None")
            if 'ViT' in trainer_name:
                base_folder = join(folder, "Generic_ViT_UNetV1", "SEQ", "head_None")

            # Look for all fold_* directories
            fold_pattern = join(base_folder, "fold_*")
            fold_folders = glob.glob(fold_pattern)

            for fold_folder in fold_folders:
                fold_name = os.path.basename(fold_folder)
                if fold_name.startswith("fold_"):
                    try:
                        fold_num = int(fold_name.split("_")[1])

                        # Initialize nested dictionaries if needed
                        if key not in experiments_dict:
                            experiments_dict[key] = {}
                        if current_tasks_path not in experiments_dict[key]:
                            experiments_dict[key][current_tasks_path] = {}

                        # Store or update fold entry
                        entry = [date_object, fold_folder]
                        if fold_num not in experiments_dict[key][current_tasks_path]:
                            experiments_dict[key][current_tasks_path][fold_num] = entry
                        elif experiments_dict[key][current_tasks_path][fold_num][0] < entry[0]:
                            experiments_dict[key][current_tasks_path][fold_num] = entry

                    except (ValueError, IndexError):
                        continue

    # Print summary
    print("\nFound experiments with folds:")
    for exp_name, task_dict in experiments_dict.items():
        print(f"\nExperiment: {exp_name}")
        for task_combination, fold_dict in task_dict.items():
            available_folds = sorted(fold_dict.keys())
            print(f"  {task_combination}: folds {available_folds}")

    return experiments_dict


def compute_btw_for_trainer_clinical_cv(trainer_name: str, tasks: List[str], evaluation_folder: str, res: str = "2d"):
    """
    Compute BTW metrics with clinical CV and D×D matrix visualization.
    """
    print(f"\nProcessing clinical CV metrics for trainer: {trainer_name}")

    # Find all experiment folders with fold information
    exp_folders = find_experiment_folders_with_folds(evaluation_folder, trainer_name, tasks, res)

    # Find sequential trainer folders for single task models
    avg_num_folds_domain_model_per_task = 0
    seq_folders = {}
    for task in tasks[1:]:
        singe_domain_exp_folders = find_experiment_folders_with_folds(evaluation_folder, "nnUNetTrainerSequential", [task], res)
        if 'seq' in singe_domain_exp_folders:
            if 'seq' not in seq_folders:
                seq_folders['seq'] = {}
            seq_folders['seq'].update(singe_domain_exp_folders['seq'])
        else:
            continue
        avg_num_folds_domain_model = 0
        for task in seq_folders['seq']:
            keys = len(seq_folders['seq'][task])
            avg_num_folds_domain_model += keys
        avg_num_folds_domain_model /= len(seq_folders['seq'])
        avg_num_folds_domain_model_per_task += avg_num_folds_domain_model

    avg_num_folds_domain_model_per_task /= len(tasks[1:])

    if not exp_folders:
        print(f"No experiment folders found for trainer {trainer_name}")
        return

    # Compute transfer metrics with clinical CV
    transfer_metrics = compute_transfer_metrics_clinical_cv(exp_folders, seq_folders, tasks)

    # Write summary with D×D matrix visualization
    summary_base_path = join(evaluation_folder, "aggregated_summaries", trainer_name)
    os.makedirs(summary_base_path, exist_ok=True)

    for exp_name in exp_folders.keys():
        task_ids_string = "_".join(tasks)

        avg_num_folds_per_task = 0
        for task in exp_folders[exp_name]:
            keys = len(exp_folders[exp_name][task])
            avg_num_folds_per_task += keys
        avg_num_folds_per_task = avg_num_folds_per_task / len(exp_folders[exp_name])

        summary_file = join(summary_base_path, f"clinical_cv_matrix_{exp_name}_{task_ids_string}.txt")
        append_clinical_cv_summary_with_matrix(summary_file, {exp_name: transfer_metrics[exp_name]}, tasks, avg_num_folds_per_task, avg_num_folds_domain_model_per_task)
        print(f"Written clinical CV matrix summary to {summary_file}")

    return transfer_metrics


def eval_task_crossvalidation(trainers, task_ids, gpu: str = "0", res: str = "2d"):
    """
    Run evaluation for datasets using cross-validation across all 5 folds, then compute aggregated transfer metrics.
    This assumes all folds have already been trained and evaluated.

    Args:
        trainers (list): List of trainer names to evaluate
        task_ids (list): List of task IDs in sequence order
        gpu (str): GPU device ID
        res (str): Resolution, default "2d"
    """
    # Set environment variables
    evaluation_folder = os.environ["EVALUATION_FOLDER"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(os.environ["EVALUATION_FOLDER"], exist_ok=True)

    for trainer in trainers:
        # Convert task IDs to task names
        tasks = [convert_id_to_task_name(task_id) for task_id in task_ids]

        print(f"\nProcessing cross-validation analysis for trainer: {trainer}")
        print(f"Tasks: {tasks}")

        # Check if we have sufficient data
        if len(tasks) < 2:
            print("Need at least 2 tasks for continual learning evaluation")
            continue

        # Compute transfer metrics using 5-fold cross validation
        print(f"Computing transfer metrics with 5-fold cross validation...")
        transfer_metrics = compute_btw_for_trainer_clinical_cv(trainer, tasks, evaluation_folder, res)

        if transfer_metrics:
            print(f"Successfully computed transfer metrics for {len(transfer_metrics)} experiments")

            # Print summary with BOTH AVG metrics including standard deviations
            for exp_name, mask_metrics in transfer_metrics.items():
                print(f"\nExperiment: {exp_name}")
                for mask_name, metrics in mask_metrics.items():
                    print(f"  {mask_name}:")
                    print(
                        f"    AVG (equal timestep): {metrics.get('avg_dice_timestep_equal', 0):.4f} ± {metrics.get('avg_dice_timestep_equal_std', 0):.4f}")
                    print(
                        f"    AVG (BWT-style): {metrics.get('avg_dice_bwt_style', 0):.4f} ± {metrics.get('avg_dice_bwt_style_std', 0):.4f}")
                    print(f"    BWT: {metrics.get('bwt_dice_mean', 0):.4f} ± {metrics.get('bwt_dice_std', 0):.4f}")
                    print(f"    FWT: {metrics.get('fwt_dice_mean', 0):.4f} ± {metrics.get('fwt_dice_std', 0):.4f}")
        else:
            print(f"No transfer metrics computed for trainer {trainer}")


def extract_and_merge_test_results_across_folds(fold_paths: Dict[int, List]) -> Union[
    dict[Any, Any], tuple[dict[Any, Any], list[Any]]]:
    """
    Extract individual test case results from all folds and merge them to compute
    domain-level performance with within-domain standard deviation.

    Args:
        fold_paths: Dictionary mapping fold numbers to [date_object, folder_path]

    Returns:
        Aggregated metrics with within-domain std based on individual test cases
    """
    all_fold_individual_results = {}
    # Extract individual test case results from each fold
    for fold_num, (date_obj, folder_path) in fold_paths.items():
        metrics_path = join(folder_path, "val_metrics_test.csv")
        if os.path.exists(metrics_path):
            fold_results = extract_individual_test_cases_from_csv(metrics_path)
            all_fold_individual_results[fold_num] = fold_results
            print(f"Extracted individual test cases from fold {fold_num}")
        else:
            print(f"Warning: No metrics file found for fold {fold_num} at {metrics_path}")

    if not all_fold_individual_results:
        print("No valid fold metrics found")
        return {}

    # Merge individual test cases across all folds for each domain
    merged_metrics = {}

    # Get all tasks and masks from the first available fold
    first_fold_results = next(iter(all_fold_individual_results.values()))
    all_tasks = set()
    all_masks = set()

    for task_name, task_results in first_fold_results.items():
        all_tasks.add(task_name)
        for mask_name in task_results.keys():
            all_masks.add(mask_name)

    # Merge individual test cases for each task and mask
    for task_name in all_tasks:
        merged_metrics[task_name] = {}

        for mask_name in all_masks:
            all_dice_cases = []
            all_masd_cases = []

            # Collect individual test cases from all folds
            for fold_num, fold_results in all_fold_individual_results.items():
                if (task_name in fold_results and
                        mask_name in fold_results[task_name]):
                    # Get individual test case scores (not aggregated)
                    fold_dice_cases = fold_results[task_name][mask_name]['individual_dice']
                    fold_masd_cases = fold_results[task_name][mask_name]['individual_masd']

                    # Check for NaN values in fold_masd_cases and replace with 0.0
                    fold_masd_cases = [0.0 if np.isnan(x) else x for x in fold_masd_cases]

                    all_dice_cases.extend(fold_dice_cases)
                    all_masd_cases.extend(fold_masd_cases)

            # Compute domain statistics from all individual test cases
            if all_dice_cases:
                merged_metrics[task_name][mask_name] = {
                    'dice': np.mean(all_dice_cases),
                    'masd': np.mean(all_masd_cases),
                    'dice_std': np.std(all_dice_cases),  # Within-domain std
                    'masd_std': np.std(all_masd_cases),  # Within-domain std
                    'n_cases': len(all_dice_cases),
                    'n_folds': len(fold_paths)
                }
                print(f"Merged {task_name} {mask_name}: "
                      f"Dice={merged_metrics[task_name][mask_name]['dice']:.4f}±{merged_metrics[task_name][mask_name]['dice_std']:.4f} "
                      f"from {len(all_dice_cases)} test cases across {len(fold_paths)} folds")

    return merged_metrics


def extract_individual_test_cases_from_csv(file_path: str) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Extract individual test case results (not aggregated) from CSV file.

    Returns:
        Dict: {task_name: {mask_name: {'individual_dice': [list], 'individual_masd': [list]}}}
    """
    individual_results = {}
    print(f"\nProcessing individual test cases from: {file_path}")

    try:
        df = pd.read_csv(file_path, sep='\t')

        # Filter for individual test cases (not aggregates)
        individual_cases = df[
            (~df['Task'].str.contains('_AGGREGATE', na=False)) &
            (~df['Task'].str.contains('ALL_TASKS', na=False)) &
            (~df['Task'].str.contains('META', na=False)) &
            (df['subject_id'] != 'average')  # Exclude averaged results
            ]

        print(f"Found {len(individual_cases)} individual test case rows")

        # Group by task and extract individual scores
        for task_name in individual_cases['Task'].unique():
            if task_name not in individual_results:
                individual_results[task_name] = {}

            task_data = individual_cases[individual_cases['Task'] == task_name]

            # Get unique subject IDs for this task
            subjects = task_data['subject_id'].unique()
            mask_names = task_data['seg_mask'].unique()

            for mask_name in mask_names:
                if mask_name not in individual_results[task_name]:
                    individual_results[task_name][mask_name] = {
                        'individual_dice': [],
                        'individual_masd': []
                    }

                # Extract individual Dice and MASD scores for each subject
                for subject_id in subjects:
                    subject_data = task_data[task_data['subject_id'] == subject_id]
                    subject_data = subject_data[subject_data['seg_mask'] == mask_name]

                    dice_row = subject_data[subject_data['metric'] == 'Dice']
                    masd_row = subject_data[subject_data['metric'] == 'MASD']

                    if not dice_row.empty and not masd_row.empty:
                        dice_score = float(dice_row.iloc[0]['value'])
                        masd_score = float(masd_row.iloc[0]['value'])

                        if np.isnan(masd_score):
                            masd_score = 0.0

                        individual_results[task_name][mask_name]['individual_dice'].append(dice_score)
                        individual_results[task_name][mask_name]['individual_masd'].append(masd_score)

                print(
                    f"Extracted {len(individual_results[task_name][mask_name]['individual_dice'])} individual cases for {task_name}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}

    return individual_results


def append_clinical_cv_summary_with_matrix(output_file: str,
                                           transfer_metrics: Dict[str, Dict[str, Dict[str, float]]],
                                           tasks: List[str],
                                           avg_num_folds_per_task: int = 0,
                                           avg_num_folds_domain_model_per_task:  int = 0):
    """
    Append transfer metrics with D×D performance matrix visualization for both Dice and MASD.
    """
    with open(output_file, 'a') as f:
        f.write("\nClinical Cross-Validation Transfer Learning Metrics:\n")
        f.write("=" * 80 + "\n")

        f.write(f"Folds that were found (average): {avg_num_folds_per_task}\n")
        f.write(f"Folds that were found for single domain models (average, needed for FWT): {avg_num_folds_domain_model_per_task}\n")

        for exp_name, mask_metrics in transfer_metrics.items():
            f.write(f"\nExperiment: {exp_name}\n")
            f.write("-" * 50 + "\n")

            for mask_name, metrics in mask_metrics.items():
                f.write(f"\n--- {mask_name.upper()} ---\n")

                # Write summary metrics first
                f.write("\nSummary Metrics:\n")
                f.write("DICE METRICS:\n")
                f.write(
                    f"  AVG (equal timestep): {metrics['avg_dice_timestep_equal']:.4f} ± {metrics['avg_dice_timestep_equal_std']:.4f}\n")
                f.write(
                    f"  AVG (BWT-style): {metrics['avg_dice_bwt_style']:.4f} ± {metrics['avg_dice_bwt_style_std']:.4f}\n")
                f.write(f"  BWT: {metrics['bwt_dice_mean']:.4f} ± {metrics['bwt_dice_std']:.4f}\n")
                f.write(f"  FWT: {metrics['fwt_dice_mean']:.4f} ± {metrics['fwt_dice_std']:.4f}\n")
                f.write("MASD METRICS:\n")
                f.write(
                    f"  AVG (equal timestep): {metrics['avg_masd_timestep_equal']:.4f} ± {metrics['avg_masd_timestep_equal_std']:.4f}\n")
                f.write(
                    f"  AVG (BWT-style): {metrics['avg_masd_bwt_style']:.4f} ± {metrics['avg_masd_bwt_style_std']:.4f}\n")
                f.write(f"  BWT: {metrics['bwt_masd_mean']:.4f} ± {metrics['bwt_masd_std']:.4f}\n")
                f.write(f"  FWT: {metrics['fwt_masd_mean']:.4f} ± {metrics['fwt_masd_std']:.4f}\n")

                # Build the matrix data
                D = len(tasks)
                matrix = {}

                # Populate matrix from domain_details
                for detail_key, detail_info in metrics.get('domain_details', {}).items():
                    parts = detail_key.split('_')
                    if len(parts) >= 3:
                        timestep = int(parts[1])
                        task_name = '_'.join(parts[2:])

                        if task_name in tasks:
                            domain_idx = tasks.index(task_name)
                            stage_idx = timestep - 1

                            matrix[(domain_idx, stage_idx)] = detail_info

                # DICE MATRIX
                f.write(f"\nDICE Performance Matrix (Dice ± Within-Domain Std):\n")
                f.write("Rows: Domains | Columns: Training Stages (Left→Right = Time→)\n")
                f.write("Only showing seen domains at each stage\n")
                f.write("\n")

                # Write matrix header
                header = "Domain".ljust(15)
                for stage in range(D):
                    header += f"Stage{stage + 1}".rjust(20)
                f.write(header + "\n")
                f.write("=" * (15 + 20 * D) + "\n")

                # Write DICE matrix rows
                for domain_idx, domain_name in enumerate(tasks):
                    row = f"{domain_name}".ljust(15)

                    for stage_idx in range(D):
                        if stage_idx < domain_idx:
                            cell = "---".rjust(20)
                        elif (domain_idx, stage_idx) in matrix:
                            data = matrix[(domain_idx, stage_idx)]
                            dice_str = f"{data['dice']:.3f}±{data['dice_std']:.3f}"
                            cell = dice_str.rjust(20)
                        else:
                            cell = "MISSING".rjust(20)

                        row += cell

                    f.write(row + "\n")

                # MASD MATRIX
                f.write(f"\nMASD Performance Matrix (MASD ± Within-Domain Std):\n")
                f.write("Rows: Domains | Columns: Training Stages (Left→Right = Time→)\n")
                f.write("Only showing seen domains at each stage\n")
                f.write("\n")

                # Write matrix header
                header = "Domain".ljust(15)
                for stage in range(D):
                    header += f"Stage{stage + 1}".rjust(20)
                f.write(header + "\n")
                f.write("=" * (15 + 20 * D) + "\n")

                # Write MASD matrix rows
                for domain_idx, domain_name in enumerate(tasks):
                    row = f"{domain_name}".ljust(15)

                    for stage_idx in range(D):
                        if stage_idx < domain_idx:
                            cell = "---".rjust(20)
                        elif (domain_idx, stage_idx) in matrix:
                            data = matrix[(domain_idx, stage_idx)]
                            # Need to get MASD values - add them to domain_details in compute function
                            if 'masd' in data and 'masd_std' in data:
                                masd_str = f"{data['masd']:.3f}±{data['masd_std']:.3f}"
                            else:
                                masd_str = "NO_MASD_DATA"
                            cell = masd_str.rjust(20)
                        else:
                            cell = "MISSING".rjust(20)

                        row += cell

                    f.write(row + "\n")

                f.write("\n")

                # Matrix usage explanation
                f.write("Matrix Usage Explanation (applies to both Dice and MASD):\n")
                f.write(
                    "• AVG (equal timestep): For each stage, averages all seen domains, then averages across stages\n")
                f.write(
                    "  - Stage 1: avg(domain1) → Stage 2: avg(domain1, domain2) → Stage 3: avg(domain1, domain2, domain3)\n")
                f.write("  - Final: avg(stage_averages)\n")
                f.write("• AVG (BWT-style): Averages all individual domain performances (all shown values)\n")
                f.write("• BWT: Compares same domain across different stages\n")
                f.write("• FWT: Compares continual learning against single-task baselines\n")

                # Write detailed BWT measurements for both metrics
                f.write(f"\nDetailed BWT Measurements:\n")
                sorted_measurements = sorted(metrics['bwt_measurements'],
                                             key=lambda x: (x['time_step'], tasks.index(x['prev_task'])))

                current_time_step = None
                for measurement in sorted_measurements:
                    if current_time_step != measurement['time_step']:
                        current_time_step = measurement['time_step']
                        f.write(f"\nAfter Stage {current_time_step + 1} (learning {measurement['current_task']}):\n")

                    f.write(f"  {measurement['prev_task']}:\n")
                    f.write(f"    DICE: {measurement['current_dice']:.4f}±{measurement['current_dice_std']:.4f} vs "
                            f"{measurement['original_dice']:.4f}±{measurement['original_dice_std']:.4f} "
                            f"(diff: {measurement['dice_diff']:.4f})\n")
                    f.write(
                        f"    MASD: {measurement.get('current_masd', 'N/A'):.4f}±{measurement.get('current_masd_std', 0):.4f} vs "
                        f"{measurement.get('original_masd', 'N/A'):.4f}±{measurement.get('original_masd_std', 0):.4f} "
                        f"(diff: {measurement.get('masd_diff', 0):.4f})\n")

                # Write FWT measurements for both metrics
                if metrics.get('fwt_dice'):
                    f.write(f"\nDetailed FWT Measurements:\n")
                    for i, task in enumerate(tasks[1:]):
                        dice_fwt = metrics['fwt_dice'][i] if i < len(metrics['fwt_dice']) else 0
                        masd_fwt = metrics['fwt_masd'][i] if i < len(metrics['fwt_masd']) else 0
                        f.write(f"{task}:\n")
                        f.write(f"  DICE FWT: {dice_fwt:.4f} (continual vs single-task)\n")
                        f.write(f"  MASD FWT: {masd_fwt:.4f} (continual vs single-task)\n")

                f.write("\n" + "=" * 50 + "\n")


def compute_transfer_metrics_clinical_cv(exp_folders: Dict[str, Dict[str, Dict[int, List]]],
                                         seq_folders: Dict[str, Dict[str, Dict[int, List]]],
                                         tasks: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute transfer metrics using clinical-focused cross validation with within-domain std.
    """
    transfer_metrics = {}
    print(f"\nTasks to process: {tasks}")

    # Get single task models with merged test results for FWT computation
    found_single_model_folds = []
    single_task_metrics = {}
    for task in tasks[1:]:
        if 'seq' in seq_folders and task in seq_folders['seq']:
            fold_paths = seq_folders['seq'][task]
            print(f"Merging test results for single task {task} across {len(fold_paths)} folds")
            single_task_metrics[task] = extract_and_merge_test_results_across_folds(fold_paths)
        else:
            print(f"No single task model found for task {task}")

    # Process each experiment
    for exp_name, task_paths in exp_folders.items():
        print(f"\nProcessing experiment: {exp_name}")
        transfer_metrics[exp_name] = {}

        # Determine available masks
        available_masks = set()
        for task_sequence_key, fold_paths in task_paths.items():
            temp_metrics = extract_and_merge_test_results_across_folds(fold_paths)
            for task_metrics in temp_metrics.values():
                available_masks.update(task_metrics.keys())

        print(f"Available masks: {available_masks}")

        # Initialize metrics structure for each mask
        for mask_name in available_masks:
            transfer_metrics[exp_name][mask_name] = {
                'bwt_measurements': [],
                'fwt_dice': [],
                'fwt_masd': [],
                'avg_timestep_dice': [],
                'avg_timestep_masd': [],
                'avg_all_dice': [],
                'avg_all_masd': [],
                'domain_details': {},  # Store detailed domain info for debugging
            }

        # Process each task sequence length
        for i in range(len(tasks)):
            current_task = tasks[i]
            current_sequence = tasks[:i + 1]
            current_sequence_key = '_'.join(current_sequence)

            if current_sequence_key not in task_paths:
                print(f"Sequence {current_sequence_key} not found in paths")
                continue

            print(f"\nProcessing sequence up to task {current_task}")

            # Get merged test results for this sequence
            current_metrics = extract_and_merge_test_results_across_folds(task_paths[current_sequence_key])

            if not current_metrics:
                print(f"No valid merged metrics for sequence {current_sequence_key}")
                continue

            # Compute AVG metrics for this timestep
            for mask_name in available_masks:
                # Collect dice and masd scores for all tasks seen so far at this timestep
                timestep_dice_scores = []
                timestep_masd_scores = []

                for seen_task in current_sequence:
                    if seen_task in current_metrics and mask_name in current_metrics[seen_task]:
                        dice_score = current_metrics[seen_task][mask_name]['dice']
                        masd_score = current_metrics[seen_task][mask_name]['masd']

                        timestep_dice_scores.append(dice_score)
                        timestep_masd_scores.append(masd_score)

                        # Also add to the list for BWT-style averaging
                        transfer_metrics[exp_name][mask_name]['avg_all_dice'].append(dice_score)
                        transfer_metrics[exp_name][mask_name]['avg_all_masd'].append(masd_score)

                        # Store domain details for debugging
                        key = f"timestep_{i + 1}_{seen_task}"
                        transfer_metrics[exp_name][mask_name]['domain_details'][key] = {
                            'dice': dice_score,
                            'dice_std': current_metrics[seen_task][mask_name]['dice_std'],
                            'masd': masd_score,
                            'masd_std': current_metrics[seen_task][mask_name]['masd_std'],
                            'n_cases': current_metrics[seen_task][mask_name]['n_cases'],
                            'n_folds': current_metrics[seen_task][mask_name]['n_folds']
                        }

                # Compute average for this timestep
                if timestep_dice_scores:
                    avg_dice_timestep = np.mean(timestep_dice_scores)
                    avg_masd_timestep = np.mean(timestep_masd_scores)

                    transfer_metrics[exp_name][mask_name]['avg_timestep_dice'].append(avg_dice_timestep)
                    transfer_metrics[exp_name][mask_name]['avg_timestep_masd'].append(avg_masd_timestep)

                    print(
                        f"Timestep {i + 1} AVG for {mask_name}: Dice={avg_dice_timestep:.4f}, MASD={avg_masd_timestep:.4f}")

            # Compute BWT: Compare each previous task's performance to its original performance
            if i > 0:  # Skip first task as it has no BWT
                for prev_idx, prev_task in enumerate(tasks[:i]):
                    prev_sequence = tasks[:prev_idx + 1]
                    prev_sequence_key = '_'.join(prev_sequence)

                    if prev_sequence_key not in task_paths:
                        print(f"Previous sequence {prev_sequence_key} not found in paths")
                        continue

                    print(f"Computing BWT for {prev_task} after learning {current_task}")
                    original_metrics = extract_and_merge_test_results_across_folds(task_paths[prev_sequence_key])

                    # Process each mask separately
                    for mask_name in available_masks:
                        if (prev_task in original_metrics and mask_name in original_metrics[prev_task] and
                                prev_task in current_metrics and mask_name in current_metrics[prev_task]):
                            bwt_measurement = {
                                'prev_task': prev_task,
                                'current_task': current_task,
                                'time_step': i,
                                'dice_diff': current_metrics[prev_task][mask_name]['dice'] -
                                             original_metrics[prev_task][mask_name]['dice'],
                                'masd_diff': current_metrics[prev_task][mask_name]['masd'] -
                                             original_metrics[prev_task][mask_name]['masd'],
                                # Store additional info for clinical interpretation
                                'current_dice': current_metrics[prev_task][mask_name]['dice'],
                                'current_dice_std': current_metrics[prev_task][mask_name]['dice_std'],
                                'original_dice': original_metrics[prev_task][mask_name]['dice'],
                                'original_dice_std': original_metrics[prev_task][mask_name]['dice_std'],
                                'current_masd': current_metrics[prev_task][mask_name]['masd'],
                                'current_masd_std': current_metrics[prev_task][mask_name]['masd_std'],
                                'original_masd': original_metrics[prev_task][mask_name]['masd'],
                                'original_masd_std': original_metrics[prev_task][mask_name]['masd_std'],
                                'current_n_cases': current_metrics[prev_task][mask_name]['n_cases'],
                                'original_n_cases': original_metrics[prev_task][mask_name]['n_cases']
                            }
                            transfer_metrics[exp_name][mask_name]['bwt_measurements'].append(bwt_measurement)
                            print(f"BWT measurement recorded for {mask_name}: Task {prev_task} after {current_task}: " +
                                  f"Dice={bwt_measurement['dice_diff']:.4f} " +
                                  f"({bwt_measurement['current_dice']:.4f}±{bwt_measurement['current_dice_std']:.4f} vs " +
                                  f"{bwt_measurement['original_dice']:.4f}±{bwt_measurement['original_dice_std']:.4f})")

                # Compute FWT using single task models
                for mask_name in available_masks:
                    if (current_task in single_task_metrics and
                            current_task in single_task_metrics[current_task] and
                            mask_name in single_task_metrics[current_task][current_task] and
                            current_task in current_metrics and
                            mask_name in current_metrics[current_task]):

                        print(f"Computing FWT for {current_task} {mask_name} using merged test results")

                        continual_dice = current_metrics[current_task][mask_name]['dice']
                        continual_dice_std = current_metrics[current_task][mask_name]['dice_std']
                        single_dice = single_task_metrics[current_task][current_task][mask_name]['dice']
                        single_dice_std = single_task_metrics[current_task][current_task][mask_name]['dice_std']

                        fwt_dice = continual_dice - single_dice

                        continual_masd = current_metrics[current_task][mask_name]['masd']
                        continual_masd_std = current_metrics[current_task][mask_name]['masd_std']
                        single_masd = single_task_metrics[current_task][current_task][mask_name]['masd']
                        single_masd_std = single_task_metrics[current_task][current_task][mask_name]['masd_std']

                        fwt_masd = continual_masd - single_masd

                        transfer_metrics[exp_name][mask_name]['fwt_dice'].append(fwt_dice)
                        transfer_metrics[exp_name][mask_name]['fwt_masd'].append(fwt_masd)

                        print(f"FWT for {current_task} {mask_name}: Dice={fwt_dice:.4f} " +
                              f"({continual_dice:.4f}±{continual_dice_std:.4f} vs {single_dice:.4f}±{single_dice_std:.4f})")
                    else:
                        print(f"Skipping FWT for {current_task} {mask_name}: No single task reference model available")

        # Calculate final summary statistics for each mask
        for mask_name in available_masks:
            mask_metrics = transfer_metrics[exp_name][mask_name]

            # Calculate BWT statistics
            all_dice_diffs = [m['dice_diff'] for m in mask_metrics['bwt_measurements']]
            all_masd_diffs = [m['masd_diff'] for m in mask_metrics['bwt_measurements']]

            if all_dice_diffs:
                mask_metrics['bwt_dice_mean'] = np.mean(all_dice_diffs)
                mask_metrics['bwt_dice_std'] = np.std(all_dice_diffs)
                mask_metrics['bwt_masd_mean'] = np.mean(all_masd_diffs)
                mask_metrics['bwt_masd_std'] = np.std(all_masd_diffs)
                print(f"\nOverall BWT metrics for {mask_name}:")
                print(f"Dice - Mean ± Std: {mask_metrics['bwt_dice_mean']:.4f} ± {mask_metrics['bwt_dice_std']:.4f}")
                print(f"MASD - Mean ± Std: {mask_metrics['bwt_masd_mean']:.4f} ± {mask_metrics['bwt_masd_std']:.4f}")
            else:
                print(f"\nNo BWT measurements available for {mask_name}")
                mask_metrics['bwt_dice_mean'] = 0.0
                mask_metrics['bwt_dice_std'] = 0.0
                mask_metrics['bwt_masd_mean'] = 0.0
                mask_metrics['bwt_masd_std'] = 0.0

            # Calculate AVG metrics - Equal timestep weighting
            if mask_metrics['avg_timestep_dice']:
                mask_metrics['avg_dice_timestep_equal'] = np.mean(mask_metrics['avg_timestep_dice'])
                mask_metrics['avg_dice_timestep_equal_std'] = np.std(mask_metrics['avg_timestep_dice'])
                mask_metrics['avg_masd_timestep_equal'] = np.mean(mask_metrics['avg_timestep_masd'])
                mask_metrics['avg_masd_timestep_equal_std'] = np.std(mask_metrics['avg_timestep_masd'])
                print(f"\nAVG metrics (equal timestep weighting) for {mask_name}:")
                print(
                    f"Dice: {mask_metrics['avg_dice_timestep_equal']:.4f} ± {mask_metrics['avg_dice_timestep_equal_std']:.4f}")
                print(
                    f"MASD: {mask_metrics['avg_masd_timestep_equal']:.4f} ± {mask_metrics['avg_masd_timestep_equal_std']:.4f}")
            else:
                mask_metrics['avg_dice_timestep_equal'] = 0.0
                mask_metrics['avg_dice_timestep_equal_std'] = 0.0
                mask_metrics['avg_masd_timestep_equal'] = 0.0
                mask_metrics['avg_masd_timestep_equal_std'] = 0.0

            # Calculate AVG metrics - BWT-style weighting (all individual scores)
            if mask_metrics['avg_all_dice']:
                mask_metrics['avg_dice_bwt_style'] = np.mean(mask_metrics['avg_all_dice'])
                mask_metrics['avg_dice_bwt_style_std'] = np.std(mask_metrics['avg_all_dice'])
                mask_metrics['avg_masd_bwt_style'] = np.mean(mask_metrics['avg_all_masd'])
                mask_metrics['avg_masd_bwt_style_std'] = np.std(mask_metrics['avg_all_masd'])
                print(f"\nAVG metrics (BWT-style weighting) for {mask_name}:")
                print(f"Dice: {mask_metrics['avg_dice_bwt_style']:.4f} ± {mask_metrics['avg_dice_bwt_style_std']:.4f}")
                print(f"MASD: {mask_metrics['avg_masd_bwt_style']:.4f} ± {mask_metrics['avg_masd_bwt_style_std']:.4f}")
            else:
                mask_metrics['avg_dice_bwt_style'] = 0.0
                mask_metrics['avg_dice_bwt_style_std'] = 0.0
                mask_metrics['avg_masd_bwt_style'] = 0.0
                mask_metrics['avg_masd_bwt_style_std'] = 0.0

            # Calculate FWT statistics
            for metric in ['fwt_dice', 'fwt_masd']:
                values = mask_metrics[metric]
                if values:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    mask_metrics[f'{metric}_mean'] = mean_value
                    mask_metrics[f'{metric}_std'] = std_value
                    print(f"\n{mask_name} {metric} summary - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
                else:
                    mask_metrics[f'{metric}_mean'] = 0.0
                    mask_metrics[f'{metric}_std'] = 0.0

    return transfer_metrics


if __name__ == "__main__":
    trainers = ["nnUNetTrainerFOCUS", "nnUNetTrainerSequential"]
    task_ids = [111, 112, 113, 115]

    for i in list(range(5)):
        eval_task(trainers, task_ids, fold=i)

    eval_task_crossvalidation(trainers, task_ids)