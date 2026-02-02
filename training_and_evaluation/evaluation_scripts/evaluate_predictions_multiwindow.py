"""
This script can be used to evaluate the predictions of a model on the sliding windows
benchmark protocol.

By default it computes the same metrics logged by the default set of metrics
in the training script, but you can add your own metrics by implementing adapting the code a bit.

Predictions will be taken from the npz files in the experiments folder.
By default, the "test" or "validate" ones will be taken, but you can change this by
by changing the _find_predictions_file function.

The output is a set of plots and a set of metrics printed to the console.
The script is designed to be run from the command line. Check the arguments of the
evaluate_multiwindow_predictions for more details!
"""

from collections import defaultdict
import re
from pathlib import Path
import sys
from typing import Literal, Union
import numpy as np
import matplotlib.pyplot as plt
import click

from evaluation_scripts.evaluate_predictions_utils import evaluate_predictions

PREDICTIONS_FILE_FORMAT = re.compile(r"predictions_test_epoch=\d+-step=(\d+)\.npz")
PREDICTIONS_FILE_FORMAT_ALT = re.compile(
    r"predictions_validate_epoch=\d+-step=(\d+)\.npz"
)


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
    default="<default_experiment_folder>",
)
@click.option(
    "--computed_on",
    type=click.Choice(["immediate_future", "growing", "growing_whole", "past", "all"]),
    default="immediate_future",
)
@click.option(
    "--evaluation_task",
    type=click.Choice(["binary", "multiclass"]),
    default="binary",
)
@click.option("--threshold", type=FLOAT_OR_AUTO, default=0.5)
@click.option("--use_pairing", type=bool, default=True)
@click.option("--results_plot_dir", type=click.Path(), default="./eval_plots")
@click.option("--save_plots", type=bool, default=False)
@click.option("--print_csv", type=bool, default=True)
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
    results_plot_dir,
    save_plots,
    print_csv,
    dataset_path,
):
    metrics_timeline = defaultdict(list)

    if save_plots:
        results_plot_dir = Path(results_plot_dir)
        results_plot_dir.mkdir(exist_ok=True, parents=True)

    experiment_path = Path(experiment_path)
    for window_id in range(9):
        print("----- Window", window_id, "-----")

        # Load the npz file
        predictions_file = _find_predictions_file(
            experiment_path / f"window_{window_id}"
        )

        window_metrics = evaluate_predictions(
            dataset_path=dataset_path,
            predictions_file=predictions_file,
            computed_on=computed_on,
            window_id=window_id,
            threshold=threshold,
            use_pairing=use_pairing,
            unknown_model_predictions_converter=convert_unknown_model_predictions,
            save_plots_dir=results_plot_dir if save_plots else None,
            evaluation_task=evaluation_task,
        )

        # Append metrics to the timeline
        for metric_name, metric_value in window_metrics.items():
            metrics_timeline[metric_name].append(metric_value)

        print(f"Overall Accuracy: {window_metrics['accuracy']:.4f}")
        print(f"Overall Balanced Accuracy: {window_metrics['balanced_accuracy']:.4f}")
        print(f"Overall Precision: {window_metrics['precision']:.4f}")
        print(f"Overall Recall: {window_metrics['recall']:.4f}")
        print(f"Overall F1 Score: {window_metrics['f1']:.4f}")
        if "tnr" in window_metrics:
            print(f"Overall True Negative Rate (TNR): {window_metrics['tnr']:.4f}")
        print(f"Average Precision: {window_metrics['average_precision']:.4f}")
        print(f"ROC AUC Score: {window_metrics['roc_auc']:.4f}")

    # Print and plot metrics timeline
    print("----- Metrics Timeline -----")
    for metric_name, metric_values in metrics_timeline.items():
        if metric_name == "counts_per_generator":
            continue

        print(f"{metric_name}: {metric_values}")

        if save_plots:
            plt.figure()
            plt.plot(metric_values, marker=".")
            plt.title(f"{metric_name} Timeline")
            plt.xlabel("Window")
            plt.ylabel(metric_name)
            plt.savefig(results_plot_dir / f"timeline_{computed_on}_{metric_name}.png")
            plt.close()

    print("----- Average Over Time -----")
    for metric_name, metric_values in metrics_timeline.items():
        if metric_name.endswith("per_generator") or metric_name.endswith(
            "per_conditioning"
        ):
            continue

        computed_on_values = metric_values
        if computed_on == "immediate_future":
            # Discard last value for immediate future
            computed_on_values = computed_on_values[:-1]

        average_value = np.mean(computed_on_values)
        print(f"{metric_name} Average: {average_value:.4f}")

    if print_csv:
        experiment_name = experiment_path.name
        values_to_print = metrics_timeline["roc_auc"]
        if computed_on == "immediate_future":
            # Discard last value for immediate future
            values_to_print = values_to_print[:-1]

        print(experiment_name)
        for value in values_to_print:
            print(f"{value:.6f}")

        # Italian locale
        print()
        print(experiment_name)
        for value in values_to_print:
            print(f"{value:.6f}".replace(".", ","))


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
