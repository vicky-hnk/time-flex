"""
Main functions to execute training and testing of Pytorch models.
"""
from src.data_loader.timeseries_data_loader import provide_series
from src.data_loader.timeseries_data_loader_cycle import provide_series_cycle
from src.data_loader.timeseries_data_loader_wavemask import provide_series_mask
from src.evaluation.evaluator import ModelEvaluator
from src.experiments.executor import LearningProcessor
from src.util.argument_parser import parse_args
from src.util.dataset_manager import (merge_params, considered_datasets,
                                      cycle_datasets, mask_datasets)
from src.util.general_utils import Attribs
from src.util.train_utils import set_seeds


def exec_shell(model_type, teacher_forcing=False):
    """Performs training and testing of the model. If desired,
    creates figures of experimental results."""
    cmd_params = vars(parse_args())  # convert Namespace to Dict
    merged_params = merge_params(cmd_params)
    params = Attribs(merged_params)
    model = model_type(params=params)
    run(model=model, params=params,
        teacher_forcing=teacher_forcing)


def run(model, params, teacher_forcing: bool = False) -> None:
    """
    Performs training and testing of the model. If desired, creates figures
    of experimental results.
    """
    if params.random_seed:
        set_seeds(params.random_seed)

    # 1-CREATE PROCESSING INSTANCE

    if any(substr in model.__class__.__name__ for substr in
           {'CycleNet'}):
        processor_class = LearningProcessor(
            provide_series_method=provide_series_cycle, model=model,
            params=params, data_dict=cycle_datasets, teacher_force=False)
    elif (any(substr in model.__class__.__name__ for substr in {'DLinear'})
          and params.aug_type is not None and params.aug_type != 0):
        processor_class = LearningProcessor(
            provide_series_method=provide_series_mask, model=model,
            params=params, data_dict=mask_datasets, teacher_force=False)
    else:
        processor_class = LearningProcessor(
            provide_series_method=provide_series, model=model,
            params=params, data_dict=considered_datasets,
            teacher_force=teacher_forcing)

    # 2-TRAINING
    if any(substr in model.__class__.__name__ for substr in
           {'CycleNet'}):
        processor_class.train_cycle()
    else:
        processor_class.train()

    # 3-TESTING
    if any(substr in model.__class__.__name__ for substr in
           {'CycleNet'}):
        pred_values, true_values, _, marks_y = processor_class.test_cycle()
    else:
        pred_values, true_values, _, marks_y = processor_class.test()

    # 4- EVALUATION
    ModelEvaluator.save_evaluation(predictions=pred_values,
                                   truth=true_values)
    if params.aug_type in {1, 2, 3, 4, 5}:
        pass
    print("Experiment run completed successfully!")
