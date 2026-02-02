"""
This script can be used to evaluate the predictions of a model on the full training
benchmark protocol.

By default it computes the same metrics logged by the default set of metrics
in the training script, but you can add your own metrics by implementing adapting the code a bit.

Predictions will be taken from the npz files in the experiments folder.
By default, the "test" or "validate" ones will be taken, but you can change this by
by changing the _find_predictions_file function.

The output is a set of plots and a set of metrics printed to the console.
The script is designed to be run from the command line.
"""

import re
from pathlib import Path
import sys
from typing import Dict
import numpy as np
import click
import matplotlib.pyplot as plt

from evaluation_scripts.evaluate_predictions_utils import evaluate_predictions
from ai_gen_bench_metadata import BENCHMARK_GENERATORS

PREDICTIONS_FILE_FORMAT = re.compile(r"predictions_test_epoch=\d+-step=(\d+)\.npz")
PREDICTIONS_FILE_FORMAT_ALT = re.compile(
    r"predictions_validate_epoch=\d+-step=(\d+)\.npz"
)

# experiment_DINOv2_tune_resize_am4_multiclass
# experiment_ViTL14_openclip_tune_resize_am4_multiclass
# experiment_RN50_clip_tune_resize_am4_multiclass


class FloatOrAuto(click.ParamType):
    name = "float|auto"

    def convert(self, value, param, ctx):
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            if value.lower() == "auto":
                return "auto"
            try:
                return float(value)
            except ValueError:
                self.fail(f"{value!r} is not a valid float or 'auto'", param, ctx)
        self.fail(f"{value!r} is not a valid input", param, ctx)


FLOAT_OR_AUTO = FloatOrAuto()


@click.command()
@click.argument(
    "experiment_path",
    type=click.Path(exists=True),
    default="/qnap_nfs/ai_gen_bench_journal_extension_exps/leonardo/experiments_data/experiment_DINOv2_tune_resize_soft_train_aug_eval_mix_am4_epochs3_full",
)
@click.option(
    "--computed_on",
    type=click.Choice(["immediate_future", "growing", "growing_whole", "past", "all"]),
    default="all",
)
@click.option(
    "--evaluation_task",
    type=click.Choice(["binary", "multiclass"]),
    default="binary",
)
@click.option("--threshold", type=FLOAT_OR_AUTO, default=0.5)
@click.option("--use_pairing", type=bool, default=True)
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    default="<default_experiment_folder>",
)
def evaluate_multiwindow_predictions(
    experiment_path,
    computed_on,
    evaluation_task,
    threshold,
    use_pairing,
    dataset_path,
):
    experiment_path = Path(experiment_path)
    window_id = 0
    print("----- Window", window_id, "-----")

    # Load the npz file
    predictions_file = _find_predictions_file(experiment_path / f"window_{window_id}")

    window_metrics = evaluate_predictions(
        dataset_path=Path(dataset_path),
        predictions_file=predictions_file,
        computed_on=computed_on,
        window_id=window_id,
        threshold=threshold,
        use_pairing=use_pairing,
        unknown_model_predictions_converter=convert_unknown_model_predictions,
        save_plots_dir=None,
        evaluation_task=evaluation_task,
    )

    print(f"Overall Accuracy: {window_metrics['accuracy']:.4f}")
    print(f"Overall Balanced Accuracy: {window_metrics['balanced_accuracy']:.4f}")
    print(f"Overall Precision: {window_metrics['precision']:.4f}")
    print(f"Overall Recall: {window_metrics['recall']:.4f}")
    print(f"Overall F1 Score: {window_metrics['f1']:.4f}")
    if "tnr" in window_metrics:
        print(f"Overall True Negative Rate (TNR): {window_metrics['tnr']:.4f}")
    print(f"Average Precision: {window_metrics['average_precision']:.4f}")
    print(f"ROC AUC Score: {window_metrics['roc_auc']:.4f}")

    generator_id_to_name = sorted(BENCHMARK_GENERATORS.keys())
    generator_id_to_date = [
        BENCHMARK_GENERATORS[gen_name] for gen_name in generator_id_to_name
    ]

    # Sort generators by date
    date_name_pairs = list(zip(generator_id_to_date, generator_id_to_name))
    date_name_pairs.sort(key=lambda x: x[0])  # Sort by date
    generators_sorted_by_date = [name for date, name in date_name_pairs]

    if "recall_per_generator" in window_metrics:
        recall_per_generator: Dict[int, float] = window_metrics["recall_per_generator"]
        print("Recall per generator:")
        for generator_id in sorted(recall_per_generator.keys()):
            if generator_id == 0:
                continue
            recall = recall_per_generator[generator_id]
            generator_name = generator_id_to_name[generator_id - 1]
            print(f"Generator {generator_name}: {recall:.4f}")

        # Plot recall (bar plot, order generator by date)
        # Use simple integer positions for x-axis instead of dates
        x_positions = range(len(generators_sorted_by_date))
        recalls = []
        for gen_name in generators_sorted_by_date:
            # Find the generator_id for this generator name
            gen_id = (
                generator_id_to_name.index(gen_name) + 1
            )  # +1 because real class is 0
            recalls.append(recall_per_generator.get(gen_id, 0.0))

        plt.figure(figsize=(12, 6))
        plt.bar(x_positions, recalls, color="blue", alpha=0.7)
        plt.xlabel("Generator (ordered by date)")
        plt.ylabel("Recall")
        plt.ylim(0.7, 1)
        plt.title("Recall per Generator")
        plt.xticks(x_positions, generators_sorted_by_date, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"eval_plots/recall_per_generator_window_{window_id}.png")
        plt.show()

    if "recall_per_conditioning" in window_metrics:
        conditioning_recalls: Dict[str, float] = window_metrics[
            "recall_per_conditioning"
        ]
        print("Recall per conditioning:")
        for conditioning in sorted(conditioning_recalls.keys()):
            recall = conditioning_recalls[conditioning]
            print(f"Conditioning {conditioning}: {recall:.4f}")

        # Plot recall per conditioning (bar plot) - sort by conditioning name
        conditionings = sorted(conditioning_recalls.keys())
        recalls = [conditioning_recalls[cond] for cond in conditionings]

        plt.figure(figsize=(12, 6))
        plt.bar(conditionings, recalls, color="blue", alpha=0.7)
        plt.xlabel("Conditioning")
        plt.ylabel("Recall")
        plt.ylim(0.7, 1)
        plt.title("Recall per Conditioning")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"eval_plots/recall_per_conditioning_window_{window_id}.png")
        plt.show()


def convert_unknown_model_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    generator_names_to_ids: np.ndarray,
) -> np.ndarray:
    print(
        "This happens because your model was trained on a different set of generators than the evaluation set",
        file=sys.stderr,
    )
    print(
        "Don't worry, just implement convert_unknown_model_predictions at the bottom of the script (where the following exception is raised)",
        file=sys.stderr,
    )
    raise ValueError(
        "You should implement a custom conversion from your model output to binary scores"
    )

    # In most cases, your model is trained to predict the generator ID directly, which means that 0 is usually the real class and the rest are fake classes
    # In that case you predictions shape is [num_samples, num_generators+1] (+1 because of the real class).

    # Ensure predictions are binary class labels (0 for real, 1 for fake)
    # Your code here ...

    # return binary_classification_scores


def _find_predictions_file(
    sliding_window_results_path: Path, allow_highest_step_file: bool = True
) -> Path:
    # The "_all" file is the last one for each window
    candidate_predictions_files = [
        sliding_window_results_path / "predictions_test_all.npz",
        sliding_window_results_path / "predictions_validate_all.npz",
    ]
    selected_file = None

    for candidate_file in candidate_predictions_files:
        if candidate_file.exists():
            selected_file = candidate_file
            break
    else:
        if allow_highest_step_file:
            # Otherwise, we take the last one (the one with the highest step)
            max_step = -1
            for file in sliding_window_results_path.glob("*.npz"):
                match = PREDICTIONS_FILE_FORMAT.match(file.name)
                match_alt = PREDICTIONS_FILE_FORMAT_ALT.match(file.name)
                if match:
                    step = int(match.group(1))
                    if step > max_step:
                        max_step = step
                        selected_file = file
                elif match_alt:
                    step = int(match_alt.group(1))
                    if step > max_step:
                        max_step = step
                        selected_file = file

    if selected_file is None:
        raise FileNotFoundError(
            f"No predictions file found in {sliding_window_results_path}"
        )

    return selected_file


if __name__ == "__main__":
    evaluate_multiwindow_predictions()
