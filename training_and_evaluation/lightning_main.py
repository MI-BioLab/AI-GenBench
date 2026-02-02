import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from large_image import LargeImage
import datasets

datasets.features.features.register_feature(LargeImage, "LargeImage")


from training_utils.lightning_cli_sliding_windows import LightningCLISlidingWindow
from typing import List, TYPE_CHECKING, Optional, Union
import submitit

import sys
from algorithms.models import *
from lightning.pytorch.trainer.states import TrainerStatus
from lightning.pytorch.loggers import WandbLogger

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


class CheckpointableTrainingSlidingWindows:
    """
    Used to support checkpointing in SLURM (using submitit) when running a multi-window training.
    This class is used to run the training in a loop, where each iteration corresponds to a sliding window.
    The class is designed to be used with the LightningCLI, which is a command-line interface for PyTorch Lightning.
    """

    def __init__(self):
        self.experiment_resume_arguments: Optional[List[str]] = None
        self.cli = None  # transient, not stored in the checkpoint
        self.trainer = None  # transient, not stored in the checkpoint
        self.ignore_sys_argv: bool = False
        self.offline_logging: Optional[bool] = False
        self.ignore_resume_config_file: bool = False

    def __call__(
        self,
        args: Optional[List[str]] = None,
        experiment_resume_arguments: Optional[List[str]] = None,
        ignore_sys_argv: bool = False,
        offline_logging: Optional[bool] = False,
        ignore_resume_config_file: bool = False,
    ):
        ignore_sys_argv = _get_env_bool_parameter(
            "GENBENCH_IGNORE_SYS_ARGV", ignore_sys_argv
        )
        offline_logging = _get_env_optional_bool_parameter(
            "GENBENCH_OFFLINE_LOGGING", offline_logging
        )
        ignore_resume_config_file = _get_env_bool_parameter(
            "GENBENCH_IGNORE_RESUME_CONFIG_FILE", ignore_resume_config_file
        )
        self.ignore_sys_argv = ignore_sys_argv
        self.offline_logging = offline_logging
        self.ignore_resume_config_file = ignore_resume_config_file

        cli_args: List[str] = [] if args is None else args
        if ignore_sys_argv:
            print(
                "Ignoring CLI arguments. If you want to pass CLI arguments, set 'ignore_sys_argv' to False."
            )
        else:
            cli_args = sys.argv[1:] + cli_args

        if experiment_resume_arguments is not None:
            self.experiment_resume_arguments = experiment_resume_arguments
        else:
            self.experiment_resume_arguments = []

        self.cli = LightningCLISlidingWindow(
            run=True,
            args=cli_args + self.experiment_resume_arguments,
            save_config_kwargs={"overwrite": True},
            offline_logging=self.offline_logging,
            ignore_resume_config_file=self.ignore_resume_config_file,
        )
        action = self.cli.subcommand  # "fit", "test", "validate", "predict"
        is_evaluation = self.cli.is_evaluation
        is_multiwindow_evaluation = self.cli.is_multiwindow_evaluation
        self._print_timeline(self.cli)

        if experiment_resume_arguments is not None:
            assert (
                experiment_resume_arguments == self.cli.experiment_resume_arguments
            ), "Experiment resume arguments are not the same as the ones passed to the CLI"

        self.experiment_resume_arguments = self.cli.experiment_resume_arguments
        assert self.experiment_resume_arguments is not None

        experiment_id: Union[int, str] = self.cli.experiment_id
        current_window: int = self.cli.window_id
        n_timeline_windows: int = len(self.cli.datamodule.windows_timeline)

        if is_multiwindow_evaluation:
            current_window = 0  # if is_multiwindow_evaluation, then the user passed None for window_id -> start from 0
            n_windows = self.cli.count_completed_windows()
            if n_windows != n_timeline_windows:
                print(
                    "Note: the evaluation will be done on",
                    n_windows,
                    "windows of the",
                    n_timeline_windows,
                    "total windows.",
                )
            else:
                print(
                    "Note: the evaluation will be done on all",
                    n_timeline_windows,
                    "windows.",
                )
        else:
            n_windows = n_timeline_windows

        # If num_sanity_val_steps was > 0 in the config, let it run the sanity check
        # in the first iteration, but not in the next ones (it's quite a waste of time...)
        cli_args += ["--trainer.num_sanity_val_steps", "0"]

        print()
        print("-" * 20, "Experiment ID:", experiment_id, "-" * 20)
        print("IMPORTANT: In case of a crash/pause/preemption, you will be able to")
        print("resume this experiment by re-executing the same command and adding")
        print("the following arguments to the command line:")
        for resume_cli_arg in self.experiment_resume_arguments:
            print(resume_cli_arg, end=" ")
        print()
        print("-" * 80)

        # Clear the CLI object used to compute the windows timeline and general config
        self.cli = None
        success = True
        executed_once = False
        while (
            (is_multiwindow_evaluation and current_window < n_windows)
            or (
                is_evaluation
                and (not is_multiwindow_evaluation)
                and (not executed_once)
            )
            or ((not is_evaluation) and current_window < n_windows)
        ) and success:
            print(f"Running '{action}' on window {current_window}")
            executed_once = True

            # Create a new CLI object (will be used to obtain the parameters for the next window)
            if self.cli is None:
                multiwindow_evaluation_parameters = []
                if is_multiwindow_evaluation:
                    multiwindow_evaluation_parameters = [
                        "--experiment_info.sliding_windows_definition.current_window",
                        str(current_window),
                    ]
                self.cli = LightningCLISlidingWindow(
                    run=True,
                    args=cli_args
                    + self.experiment_resume_arguments
                    + multiwindow_evaluation_parameters,
                    save_config_kwargs={"overwrite": True},
                    offline_logging=self.offline_logging,
                    ignore_resume_config_file=self.ignore_resume_config_file,
                )

                assert (
                    experiment_id == self.cli.experiment_id
                ), f"Experiment ID mismatch. Expected {experiment_id}, got {self.cli.experiment_id}"
                assert (
                    current_window == self.cli.window_id
                ), f"Window ID mismatch. Expected {current_window}, got {self.cli.window_id}"

            # Keep track of the trainer: it's needed for the checkpointing
            # and for the logger
            self.trainer = self.cli.trainer

            # Actually runs the action (fit, test, validate, predict)
            self.cli.proceed_with_subcommand()

            success = self.trainer.state.status == TrainerStatus.FINISHED

            # If training, we need to mark the current window as completed.
            # Needed to make it clear that the current window is done and should not be resumed
            # as it were paused due to preemption.
            if success:
                if self.cli.is_evaluation:
                    print(
                        "Evaluation for window",
                        current_window,
                        "finished successfully.",
                    )
                else:
                    print(
                        "Training for window",
                        current_window,
                        "finished successfully.",
                    )
                    self.cli.mark_current_window_done(self.trainer.global_step)
                current_window += 1

                has_wandb = False
                for logger in self.trainer.loggers:
                    logger.finalize("success")
                    if isinstance(logger, WandbLogger):
                        has_wandb = True

                if has_wandb:
                    try:
                        import wandb

                        wandb.finish(exit_code=0)
                    except Exception as e:
                        print("Error while finishing wandb:", e)
                        pass

                # Release references
                self.cli = None
                self.trainer = None
            else:
                print(
                    "Window",
                    current_window,
                    "did not finish successfully. Exiting sliding windows loop.",
                )

        print("Window loop finished with success =", success)

    def checkpoint(
        self, args=None, experiment_resume_arguments=None
    ) -> submitit.helpers.DelayedSubmission:
        print("Got checkpoint signal.")
        print("The default root dir is", self.trainer.default_root_dir)

        if self.cli is not None and self.cli.is_evaluation:
            pass
        else:
            if self.trainer is not None:
                hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(
                    self.trainer.default_root_dir
                )
                self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer is not None:
            # Save to make sure we get all the metrics
            for logger in self.trainer.loggers:
                logger.finalize("finished")

        if experiment_resume_arguments is None:
            experiment_resume_arguments = self.experiment_resume_arguments

        self.cli = None  # Prevent the CLI from being pickled
        self.trainer = None  # Prevent the trainer from being pickled
        self.experiment_resume_arguments = None  # No need to keep this

        return submitit.helpers.DelayedSubmission(
            self,
            args=args,
            experiment_resume_arguments=experiment_resume_arguments,
            ignore_sys_argv=self.ignore_sys_argv,
            offline_logging=self.offline_logging,
            ignore_resume_config_file=self.ignore_resume_config_file,
        )  # submits to requeuing

    @staticmethod
    def _print_timeline(cli: LightningCLISlidingWindow):
        datamodule: "DeepfakeDetectionDatamodule" = cli.datamodule
        windows: List[List[str]] = (
            datamodule.windows_timeline
        )  # Each sub-list is the list of generator in that window
        n_windows = len(windows)

        print("Will run training across", n_windows, "sliding windows")
        print("Generators order:")
        for i, window in enumerate(windows):
            print(i, window)


def _get_env_bool_parameter(parameter_name: str, default_value: bool = False) -> bool:
    """
    Get an environment variable as a boolean, or return the default value if not set.
    """
    value = os.environ.get(parameter_name, str(default_value)).lower()
    return value in {"true", "1", "yes", "y"}


def _get_env_optional_bool_parameter(
    parameter_name: str, default_value: Optional[bool]
) -> Optional[bool]:
    """
    Get an environment variable as an optional boolean, or return the default value if not set.
    """
    value = os.environ.get(parameter_name, None)
    if value is None:
        return default_value
    return value.lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    CheckpointableTrainingSlidingWindows()()
