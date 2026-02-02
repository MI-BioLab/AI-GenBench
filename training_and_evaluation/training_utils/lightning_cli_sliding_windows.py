import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from pathlib import Path
import warnings

import jsonargparse
import torch

from training_utils.save_config_callback_sliding_windows import (
    SaveConfigCallbackSlidingWindows,
)
from training_utils.sliding_windows_experiment_data import (
    SlidingWindowsDefinition,
    SlidingWindowsExperimentInfo,
)


from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    ArgsType,
    SaveConfigCallback,
)

from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
from jsonargparse import Namespace


class TrainingWindowInfo(NamedTuple):
    window_id: int
    latest_checkpoint_path: Optional[Path]
    should_resume: bool


class EvaluationWindowInfo(NamedTuple):
    window_id: int
    latest_checkpoint_path: Optional[Path]
    is_done: bool


class LightningCLISlidingWindow(LightningCLI):
    """
    A patched version of LightningCLI that manages the sliding window protocol.

    Training on each sliding window is implemented as a new Lightning training. Only the previous model checkpoint is
    loaded. Each sliding window will thus create its own experiment (logging) folder, with its checkpoint folder, logs,
    predictions dump, etc.

    To manage this, some thinkering is needed to keep track the experiment ID and the window ID.
    This CLI implementation also takes care to resume the training from the last checkpoint in case of preemption or CTRL-C.

    Note: upon creating the CLI object, the `subcommand` is not executed immediately. This differs from the default behavior of
    LightningCLI, which executes the subcommand immediately.
    """

    def __init__(
        self,
        *args,
        save_config_callback: Optional[
            type[SaveConfigCallback]
        ] = SaveConfigCallbackSlidingWindows,
        offline_logging: Optional[bool] = None,
        ignore_resume_config_file: bool = False,
        **kwargs,
    ):
        self.offline_logging: Optional[bool] = offline_logging
        self.ignore_resume_config_file: bool = ignore_resume_config_file

        self._patched: bool = False
        self.is_evaluation: bool = False
        self.is_multiwindow_evaluation: bool = False
        self.experiment_info: SlidingWindowsExperimentInfo = (
            SlidingWindowsExperimentInfo()
        )
        self.sliding_windows_definition: SlidingWindowsDefinition = (
            self.experiment_info.sliding_windows_definition
        )
        self.experiments_data_folder: Path = (
            self.experiment_info.experiments_data_folder
        )
        self.logging_initial_step: int = 0
        self.evaluation_window_info: EvaluationWindowInfo = EvaluationWindowInfo(
            0, None, False
        )
        self.training_window_info: TrainingWindowInfo = TrainingWindowInfo(
            0, None, False
        )

        self.experiment_id: Union[int, str] = 0
        """
        The ID of the current experiment (can be an int or a string)
        """

        self.window_id: int = 0
        """
        The ID of the current window. This is the ID of the window 
        that is being trained or evaluated.
        """

        self.experiment_folder: Optional[Path] = None
        self.window_folder: Optional[Path] = None

        self.experiment_resume_arguments: Optional[List[str]] = None
        """
        The arguments used to resume the experiment. This is a list of strings
        (command line arguments).
        """

        self.experiment_parameters_config_file: Optional[Path] = None
        """
        The config file used to resume the experiment. This file contains the 
        loggers config, experiment_id, window_id, etc.
        It is stored in the self.window_folder.

        Note: you don't need this to resume the experiment. You can just pass
        "--experiment_info.experiment_id <exp_id>" (the exact CLI arguments are 
        stored in self.experiment_resume_arguments)
        """

        # Internal variable to manage on-the-fly config overrides
        self._resume_overrides: Optional[List[str]] = None
        self._logging_overrides: Union[Dict, Namespace] = dict()
        self._sliding_window_overrides: Union[Dict, Namespace] = dict()
        self._checkpoint_overrides: Union[Dict, Namespace] = dict()
        self._plugin_overrides: Union[Dict, Namespace] = dict()

        super().__init__(*args, save_config_callback=save_config_callback, **kwargs)

    def _run_subcommand(self, *args, **kwargs) -> None:
        if not self._patched:
            return

        return super()._run_subcommand(*args, **kwargs)

    def proceed_with_subcommand(self) -> None:
        self._patched = True

        if self.subcommand is not None:
            assert self.window_folder is not None
            self.window_folder.mkdir(parents=True, exist_ok=True)
            if not self.is_evaluation:
                self.save_experiment_parameters()  # To resume the experiment in case of CTRL-C or preemption

            return self._run_subcommand(self.subcommand)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        super().parse_arguments(parser, args)
        # Obtain relevant information from the config

        self.subcommand = self.config.subcommand
        self.is_evaluation = self.subcommand in {"validate", "test", "predict"}
        self.experiment_info = self.config[self.subcommand].experiment_info
        self.sliding_windows_definition = (
            self.experiment_info.sliding_windows_definition
        )
        self.experiments_data_folder = self.experiment_info.experiments_data_folder
        experiment_name_prefix = self.experiment_info.experiment_name_prefix
        window_name_prefix = self.experiment_info.window_name_prefix

        experiment_id = self.experiment_info.experiment_id

        # Define the experiment ID
        if experiment_id is None:
            # This is the most common case, where the experiment ID is not set
            # and we need to create a new one.
            # The experiment ID is set to the next available ID or the SLURM job ID
            # if running on SLURM. Note that this is already the default behavior of LightningCLI,
            # nothing fancy here.
            experiment_id = self._get_next_experiment_id(
                save_dir=self.experiments_data_folder,
                experiment_name_prefix=experiment_name_prefix,
            )
        self.experiment_id = experiment_id
        assert isinstance(
            self.experiment_id, (int, str)
        ), f"Experiment ID must be an int or a str, but got {type(self.experiment_id)}"
        # print("Will use experiment ID:", self.experiment_id)

        experiment_folder_name = f"{experiment_name_prefix}{self.experiment_id}"
        self.experiment_folder = (
            Path(self.experiments_data_folder) / experiment_folder_name
        )
        # print(
        #     "Experiment folder:", str(self.experiment_folder)
        # )
        # print(
        #     "Note: the experiment folder is not the same as the logging folder!"
        # )

        requested_window_id = self.sliding_windows_definition.current_window
        # Note: if requested_window_id is None, it means that the user did not set it
        # in the config file. In this case, autodetection will be used to determine
        # the current/next window ID.

        self.evaluation_window_info, self.training_window_info = (
            self._get_current_window_info(
                self.experiment_folder,
                window_name_prefix=window_name_prefix,
                configured_window_id=requested_window_id,
                check_training_consistency=not self.is_evaluation,
            )
        )

        if self.is_evaluation and requested_window_id is None:
            self.is_multiwindow_evaluation = True

        if self.is_evaluation:
            has_explicit_weights = (
                self.config[self.subcommand].ckpt_path is not None
                or self.config[self.subcommand].model.init_args.base_weights is not None
            )
            self.window_id = self.evaluation_window_info.window_id
            if (
                self.evaluation_window_info.latest_checkpoint_path is None
                and self.config[self.subcommand].ckpt_path is None
                and self.config[self.subcommand].model.init_args.base_weights is None
            ):
                raise RuntimeError(
                    f"No checkpoint was found for evaluation and not set in the CLI (wither via --chkpt_path or --model.base_weights). "
                )
            elif (not self.evaluation_window_info.is_done) and (
                not has_explicit_weights
            ):
                warnings.warn(
                    f"The window being evaluated is not done (didn't complete its epochs)! "
                    f"The 'latest' checkpoint will be evaluated, but this is not what you may expect."
                )
        else:
            self.window_id = self.training_window_info.window_id

        assert isinstance(
            self.window_id, int
        ), f"Window ID must be an int, but got {type(self.window_id)}"
        assert self.window_id >= 0

        window_folder_name = f"{window_name_prefix}{self.window_id}"
        self.window_folder = self.experiment_folder / window_folder_name
        # print("Window folder:", str(self.window_folder))

        logging_initial_step = self.experiment_info.logging_initial_step
        if logging_initial_step is not None:
            self.logging_initial_step = logging_initial_step
        else:
            if self.is_evaluation:
                from_which_checkpoint = (
                    self.evaluation_window_info.latest_checkpoint_path
                )
                if from_which_checkpoint is None:
                    from_which_checkpoint = self.config[self.subcommand].ckpt_path

                if from_which_checkpoint is None:
                    from_which_checkpoint = self.config[
                        self.subcommand
                    ].model.init_args.base_weights

                if from_which_checkpoint is None:
                    self.logging_initial_step = 0
                    warnings.warn("Could not determine the logging initial step.")
                else:
                    try:
                        self.logging_initial_step = (
                            self.get_logging_initial_step_from_checkpoint(
                                Path(from_which_checkpoint)
                            )
                        )
                    except Exception as e:
                        warnings.warn(
                            "Could not determine the logging initial step from the checkpoint.",
                        )
                        self.logging_initial_step = 0
            else:
                self.logging_initial_step = self.get_logging_initial_step(
                    self.experiment_folder,
                    self.window_id,
                    window_name_prefix=window_name_prefix,
                )

        print("Will start logging from step:", self.logging_initial_step)

        # The arguments stored in experiment_resume_arguments can be passed to the
        # CLI to resume a previously interrupted training (also printed in the main script).
        self.experiment_resume_arguments = [
            "--experiment_info.experiment_id",
            str(self.experiment_id),
        ]

        self.experiment_parameters_config_file = (
            self.find_experiment_parameters_config()
        )

        if self.experiment_parameters_config_file is not None:
            # Uses the config file to resume the experiment
            # (contains the loggers config, experiment_id, window_id, etc.)
            # For instance, it will contain the "id" of the W&B run to resume.
            if self.ignore_resume_config_file:
                print(
                    "Ignoring the experiment parameters config file at",
                    str(self.experiment_parameters_config_file),
                    "because ignore_resume_config_file is set to True.",
                )
            else:
                self._resume_overrides = [
                    "--config",
                    str(self.experiment_parameters_config_file),
                ]

        self._logging_overrides = self.make_loggers_parameters()

        self._sliding_window_overrides = {
            "experiment_info": {
                "sliding_windows_definition": {
                    "current_window": self.window_id,
                },
                "experiment_id": self.experiment_id,
                "logging_initial_step": self.logging_initial_step,
            },
            "checkpoint_callback": {
                "dirpath": str(self.window_folder / "checkpoints"),
            },
        }

        self._plugin_overrides = self.make_plugins_parameters()

        if self.is_evaluation:
            self._setup_config_for_evaluation(parser, args)
        else:
            self._setup_config_for_training(parser, args)

        # Setup config save callback
        self.save_config_kwargs["config_save_dir"] = str(self.window_folder)
        self.save_config_kwargs["config_filename"] = f"{self.subcommand}_config.yaml"

    def _setup_config_for_training(
        self, parser: LightningArgumentParser, args: ArgsType
    ) -> None:
        """
        Setups the configuration (self.config field) for training.

        This means:
        - Detecting whether the training should be resumed or not
        - Setting the experiment ID and window ID
        - Setting the proper experiment names (and other data) in loggers
        - Setting the path of the checkpoint to resume
        """
        assert self.window_folder is not None

        # Resume loggers parameters, if any
        if (
            self.training_window_info.should_resume
            and self.experiment_parameters_config_file is None
        ):
            warnings.warn(
                "Warning: no experiment parameters file found while resuming. "
                "The loggers may not be resumed properly."
            )

        self._checkpoint_overrides = dict()

        if self.training_window_info.should_resume:
            user_set_checkpoint = self.config[self.subcommand].ckpt_path
            # Resume interrupted training (because was CTRL+C-ed or preempted by SLURM)
            if user_set_checkpoint is None:
                self._checkpoint_overrides["ckpt_path"] = str(
                    self.training_window_info.latest_checkpoint_path
                )

                print(
                    "Will load the checkpoint",
                    str(self.training_window_info.latest_checkpoint_path),
                    "... as the window is being resumed from a previously interrupted training.",
                )
            else:
                print(
                    "Will load the given checkpoint",
                    str(user_set_checkpoint),
                )
        elif self.training_window_info.latest_checkpoint_path is not None:
            user_set_base_weights = self.config[
                self.subcommand
            ].model.init_args.base_weights
            if user_set_base_weights is None:
                if self.experiment_info.load_weights_from_previous_window:
                    # Checkpoint of the previous window
                    # Reload those weights for trasfer learning to the next window
                    self._checkpoint_overrides["model"] = {
                        "init_args": {
                            "base_weights": str(
                                self.training_window_info.latest_checkpoint_path
                            ),
                        },
                    }
                    print(
                        "Will load the model weights from the checkpoint of the previous window at",
                        str(self.training_window_info.latest_checkpoint_path),
                        "because previous the window finished. Those weights will be used for transfer learning.",
                    )
                else:
                    print(
                        "The previous window finished, but the model weights will not be loaded because "
                        "experiment_info.load_weights_from_previous_window is set to False."
                    )
            else:
                print(
                    "Will load the model weights from the given checkpoint at",
                    str(user_set_base_weights),
                )

        cli_args: List[str] = []
        override_dicts: List[Union[Dict, Namespace]] = [
            self._checkpoint_overrides,
            self._plugin_overrides,
        ]

        if self._resume_overrides is not None:
            cli_args.extend(self._resume_overrides)
        else:
            override_dicts.extend(
                (self._logging_overrides, self._sliding_window_overrides)
            )

        self.config = self._merge_configs(
            parser,
            variable_args=args,
            cli_args=cli_args,
            namespaces=override_dicts,
            subcommand=self.subcommand,
            validate=True,
        )

    def _setup_config_for_evaluation(
        self, parser: LightningArgumentParser, args: ArgsType
    ) -> None:
        """
        Setups the configuration (self.config field) for evaluation.

        This means:
        - Setting the experiment ID and window ID
        - Setting the proper experiment names (and other data) in loggers
        - Setting the path of the checkpoint to evaluate
        """
        assert self.window_folder is not None

        # Resume loggers parameters, if any
        if self.experiment_parameters_config_file is None:
            warnings.warn(
                "Warning: no experiment parameters file found while resuming. "
                "The loggers may not be resumed properly."
            )

        is_implicit_latest_checkpoint = False
        user_set_base_weights = self.config[self.subcommand].ckpt_path
        if user_set_base_weights is None:
            user_set_base_weights = self.config[
                self.subcommand
            ].model.init_args.base_weights
        if user_set_base_weights is None:
            user_set_base_weights = self.evaluation_window_info.latest_checkpoint_path
            is_implicit_latest_checkpoint = True

        if user_set_base_weights is None:
            raise RuntimeError(
                f"Window folder {self.window_folder} exists, but no checkpoint was found for evaluation."
            )

        if is_implicit_latest_checkpoint:
            self._checkpoint_overrides = {
                "model": {
                    "init_args": {
                        "base_weights": str(user_set_base_weights),
                    },
                }
            }
        print(
            "Will evaluate the model at",
            str(user_set_base_weights),
        )

        cli_args: List[str] = []
        override_dicts: List[Union[Dict, Namespace]] = [
            self._checkpoint_overrides,
            self._plugin_overrides,
        ]

        if self._resume_overrides is not None:
            cli_args.extend(self._resume_overrides)
        else:
            override_dicts.extend(
                (self._logging_overrides, self._sliding_window_overrides)
            )

        self.config = self._merge_configs(
            parser,
            variable_args=args,
            cli_args=cli_args,
            namespaces=override_dicts,
            subcommand=self.subcommand,
            validate=True,
        )

    @staticmethod
    def _merge_configs(
        parser: LightningArgumentParser,
        base_config: Optional[Namespace] = None,
        variable_args: ArgsType = None,
        cli_args: Optional[List[str]] = None,
        namespaces: Optional[List[Union[Dict, Namespace]]] = None,
        subcommand: Optional[str] = None,
        validate: bool = True,
    ) -> Namespace:
        """
        Merges the base config with the CLI arguments and the namespaces.
        """
        config = base_config

        # Manage the base CLI args
        # Those are of type ArgsType, which forces for a complex management
        if isinstance(variable_args, (dict, Namespace)):
            # programmatic args
            config = LightningCLISlidingWindow._merge_dict_conf(
                parser, variable_args, config
            )
        elif isinstance(variable_args, list):
            # command-line args
            cli_args = [] if cli_args is None else cli_args
            cli_args = variable_args + cli_args

        # Merge the base config with the CLI arguments
        if cli_args is not None and len(cli_args) > 0:
            config = parser.parse_args(cli_args, config)

        # Merge the config with the namespaces
        if namespaces is not None:
            for namespace in namespaces:
                if subcommand is not None and not subcommand in namespace:
                    # If the subcommand is not in the namespace, add it
                    namespace = {subcommand: namespace}

                config = LightningCLISlidingWindow._merge_dict_conf(
                    parser, namespace, config
                )

        if config is None:
            config = parser.get_defaults()

        config = parser.parse_object(config)

        if validate:
            parser.validate(config)

        return config

    @staticmethod
    def _merge_dict_conf(
        parser: LightningArgumentParser,
        from_conf: Union[Namespace, Dict],
        to_conf: Optional[Namespace],
    ) -> Namespace:
        if not isinstance(from_conf, Namespace):
            from_conf = jsonargparse.dict_to_namespace(from_conf)

        if to_conf is None:
            return from_conf

        if not isinstance(to_conf, Namespace):
            to_conf = jsonargparse.dict_to_namespace(to_conf)

        # Merge the two configs
        return parser.merge_config(from_conf, to_conf)

    def add_arguments_to_parser(self, parser):
        # Adds the experiment_info data to the parser
        parser.add_argument(
            "--experiment_info",
            type=SlidingWindowsExperimentInfo,
            default=SlidingWindowsExperimentInfo(),
        )

        # --dataset_path is linked to the data.init_args.dataset_loader.init_args.dataset_path
        # (it's easier to use this way)
        parser.add_argument(
            "--dataset_path",
            type=Optional[Path],
            default=None,
        )
        parser.add_lightning_class_args(
            ModelCheckpoint,
            "checkpoint_callback",
        )

        # Arguments linking
        parser.link_arguments(
            "experiment_info.sliding_windows_definition",
            "data.init_args.sliding_windows_definition",
        )
        parser.link_arguments(
            "experiment_info.logging_initial_step",
            "model.init_args.logging_initial_step",
        )
        parser.link_arguments(
            (
                "dataset_path",
                "data.init_args.dataset_loader.init_args.dataset_path",
            ),
            "data.init_args.dataset_loader.init_args.dataset_path",
            compute_fn=_link_dataset_path,
        )

    def mark_current_window_done(self, training_global_step: int) -> None:
        """
        Adds a marker to flag the training of the current window as completed.

        This flag will be used, when resuming an experiment, to determine whether the
        current window is done or not. If a window is done, the successive window
        will be considered for resuming the training.
        """
        if not self.trainer.strategy.is_global_zero:
            return

        assert self.window_folder is not None
        self.window_folder.mkdir(parents=True, exist_ok=True)

        with open(self.window_folder / "step_tracker", "w") as f:
            f.write(str(self.logging_initial_step + training_global_step))

        (
            self.window_folder / "done"
        ).touch()  # Creates an empty file named "done" to mark the window as done

    def current_window_latest_checkpoint(self) -> Optional[Path]:
        """
        Returns the path to the latest checkpoint of the current window.

        Note: for each window more than a checkpoint can be created, but only the last one
        is used for resuming the training. The other checkpoints are kept for reference.

        In LightningCLI the checkpoints are saved after each validation epoch (can be customized).
        """
        if self.window_folder is None:
            return None

    @staticmethod
    def _get_next_experiment_id(
        save_dir: Union[str, Path],
        experiment_name_prefix: Optional[str] = "experiment_",
    ) -> int:
        """
        Determines the next experiment ID to use.
        The experiment ID is determined by looking for the next available incremental ID in the
        experiment folder (or by looking for the SLURM job ID if running on SLURM).

        This implements the default behavior of LightningCLI.
        """
        from lightning.pytorch.plugins.environments import SLURMEnvironment
        from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

        if SLURMEnvironment.detect():
            job_id = SLURMEnvironment.job_id()
            assert job_id is not None
            return job_id

        _fs = get_filesystem(save_dir)

        try:
            listdir_info = _fs.listdir(save_dir)
        except OSError:
            warnings.warn(f"Missing logger folder: {save_dir}")
            return 0  # Default experiment id

        existing_experiments = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if _is_dir(_fs, d) and bn.startswith(experiment_name_prefix):
                dir_ver = bn.split("_")[1].replace("/", "")
                if dir_ver.isdigit():
                    existing_experiments.append(int(dir_ver))
        if len(existing_experiments) == 0:
            return 0

        return max(existing_experiments) + 1

    def _make_window_name(
        self,
        window_id: int,
        window_name_prefix: Optional[str] = "window_",
    ) -> str:
        """
        Creates the name of the window folder.
        """
        return f"{window_name_prefix}{window_id}"

    def _find_window_folders(
        self,
        experiment_path: Path,
        window_name_prefix: Optional[str] = "window_",
    ) -> List[Tuple[Path, int]]:
        """
        Finds all the window folders in the given experiment path.
        The window folders are sorted by their ID (descending).
        """

        window_folders = list(experiment_path.glob(f"{window_name_prefix}*"))
        # Order (descending) by number
        window_folders.sort(key=lambda x: int(x.name.split("_")[1]), reverse=True)

        result = list()
        for window_folder in window_folders:
            result.append((window_folder, int(window_folder.name.split("_")[1])))
        return result

    def _make_checkpoints_folder_path(
        self,
        experiment_path: Path,
        window_id: int,
        window_name_prefix: Optional[str] = "window_",
    ):
        """
        Creates the path to the checkpoints folder of the given window.
        """
        window_folder = experiment_path / self._make_window_name(
            window_id,
            window_name_prefix=window_name_prefix,
        )
        return window_folder / "checkpoints"

    def _get_window_info(
        self,
        experiment_path: Path,
        window_id: int,
        allow_prev_eval_window: bool,
        window_name_prefix: str,
        check_training_consistency: bool = True,
    ) -> Tuple[EvaluationWindowInfo, TrainingWindowInfo]:
        training_window_id = window_id
        evaluation_window_id = window_id
        del window_id

        training_window_folder = experiment_path / self._make_window_name(
            training_window_id,
            window_name_prefix=window_name_prefix,
        )
        eval_window_folder = training_window_folder

        latest_training_checkpoint = (
            LightningCLISlidingWindow.find_latest_checkpoint_file(
                self._make_checkpoints_folder_path(
                    experiment_path,
                    training_window_id,
                    window_name_prefix=window_name_prefix,
                )
            )
        )
        latest_evaluation_checkpoint = latest_training_checkpoint

        # There may be a case in which the training window exists, but no checkpoints can be found.
        # This happens if the training is interrupted before the first checkpoint is created.
        # In this case, we can use the previous window for evaluation (only if allow_prev_eval_window is True,
        # which happens only if the user didn't set a window_id in the configuration).
        if (
            latest_evaluation_checkpoint is None
            and allow_prev_eval_window
            and evaluation_window_id > 0
        ):
            evaluation_window_id -= 1
            eval_window_folder = experiment_path / self._make_window_name(
                evaluation_window_id,
                window_name_prefix=window_name_prefix,
            )
            latest_evaluation_checkpoint = (
                LightningCLISlidingWindow.find_latest_checkpoint_file(
                    self._make_checkpoints_folder_path(
                        experiment_path,
                        evaluation_window_id,
                        window_name_prefix=window_name_prefix,
                    )
                )
            )

        evaluation_window_info = EvaluationWindowInfo(
            evaluation_window_id,
            latest_evaluation_checkpoint,
            (eval_window_folder / "done").exists(),
        )

        # Compute the training window info
        if (training_window_folder / "done").exists():
            # This window is done -> move to the next one
            if check_training_consistency and latest_training_checkpoint is None:
                raise RuntimeError(
                    f"Window folder {training_window_folder} exists and is marked as completed, but no checkpoint was found. "
                    f"This should not happen. Please check your experiment folder."
                )

            training_window_info = TrainingWindowInfo(
                training_window_id + 1, latest_training_checkpoint, False
            )
        elif latest_training_checkpoint:
            # This window is not done, but we have a checkpoint.
            # This means that the training was interrupted, either by the user or
            # by the job scheduler (SLURM preemption).
            # We can resume the training from the latest checkpoint.
            training_window_info = TrainingWindowInfo(
                training_window_id, latest_training_checkpoint, True
            )
        else:
            if training_window_id == 0:
                # The first window of the experiment, we can start from scratch
                training_window_info = TrainingWindowInfo(
                    training_window_id, None, False
                )
            else:
                # The window folder exists, but it's not completed and there is no checkpoint.
                # This means that the window it's starting from scratch (or it was interrupted
                # before the first checkpoint was created).
                # We can start the training from scratch (by loading the model weights from the previous window).
                previous_window_id = training_window_id - 1
                previous_window_checkpoint = (
                    LightningCLISlidingWindow.find_latest_checkpoint_file(
                        self._make_checkpoints_folder_path(
                            experiment_path,
                            previous_window_id,
                            window_name_prefix=window_name_prefix,
                        )
                    )
                )
                # Note: it may also happend that previous_window_checkpoint is None, but
                # the window folder exists. This should not happen because a window folder is created only
                # when the previous window is done... this may be the result of the user
                # deleting some folders (or some other unexpected issue).
                if check_training_consistency and previous_window_checkpoint is None:
                    raise RuntimeError(
                        f"Window folder {training_window_folder} exists, but no checkpoint was found for either that window or the previous one. "
                        f"This should not happen. Please check your experiment folder."
                    )

                training_window_info = TrainingWindowInfo(
                    training_window_id, previous_window_checkpoint, False
                )

        return evaluation_window_info, training_window_info

    def _get_current_window_info(
        self,
        experiment_path: Path,
        window_name_prefix: str = "window_",
        configured_window_id: Optional[int] = None,
        check_training_consistency: bool = True,
    ) -> Tuple[
        EvaluationWindowInfo, TrainingWindowInfo
    ]:  # The last bool is wether the checkpoint should be loaded completely or the model weights only (True = load completely)
        """
        Returns the current window ID, the path to the latest checkpoint of the current window, and whether
        the training should be resumed from the latest checkpoint or not (that is, whether the window is done or not).
        """
        window_id = configured_window_id

        # Allow the evaluation window to be a previous one if no checkpoints are found in the current "training" window.
        allow_prev_eval_window = window_id is None
        if window_id is None:
            window_folders = self._find_window_folders(
                experiment_path,
                window_name_prefix=window_name_prefix,
            )

            if len(window_folders) == 0:
                # No window folder found, this is the first window of the experiment
                window_id = 0
            else:
                latest_window_folder = window_folders[0]
                window_folder, window_id = latest_window_folder

        return self._get_window_info(
            experiment_path,
            window_id,
            allow_prev_eval_window=allow_prev_eval_window,
            window_name_prefix=window_name_prefix,
            check_training_consistency=check_training_consistency,
        )

    def _get_completed_windows(
        self,
        experiment_path: Path,
        window_name_prefix: str = "window_",
    ) -> List[Tuple[Path, int]]:
        """
        Returns the list of completed windows (i.e. the ones that have a "done" file).
        """
        window_folders = self._find_window_folders(
            experiment_path,
            window_name_prefix=window_name_prefix,
        )

        completed_windows = list()
        for window_folder, window_id in window_folders:
            if (window_folder / "done").exists():
                completed_windows.append((window_folder, window_id))

        return completed_windows

    def count_completed_windows(
        self,
        window_name_prefix: str = "window_",
        check_consistency: bool = True,
    ) -> int:
        """
        Returns the number of completed windows (i.e. the ones that have a "done" file).
        """
        if self.experiment_folder is None:
            raise RuntimeError(
                "Experiment folder is not set. This happens if the configuration is not parsed yet."
            )

        completed_windows = self._get_completed_windows(
            self.experiment_folder,
            window_name_prefix=window_name_prefix,
        )

        n_completed_windows = len(completed_windows)

        if check_consistency:
            # Check that the completed windows are in order
            for expected_id, (window_folder, window_id) in zip(
                reversed(range(n_completed_windows)), completed_windows
            ):
                if window_id != expected_id:
                    raise RuntimeError(
                        f"Window folder {window_folder} is marked as completed, but its ID is {window_id} instead of {expected_id}. "
                        f"This should not happen. Please check your experiment folder."
                    )
        return n_completed_windows

    @staticmethod
    def find_latest_checkpoint_file(candidate_checkpoint_dir: Path) -> Optional[Path]:
        """
        Finds the latest (by timestamp) checkpoint file in the given directory (non-recursive).
        """
        if not candidate_checkpoint_dir.exists():
            return None

        checkpoint_files = list(candidate_checkpoint_dir.glob("*.ckpt"))
        if len(checkpoint_files) == 0:
            return None

        # Order by creation time
        checkpoint_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        return checkpoint_files[0]

    def get_logging_initial_step(
        self,
        experiment_path: Path,
        window_id: int,
        window_name_prefix: str = "window_",
        strict: bool = True,
    ) -> int:
        """
        Returns the initial step for logging in the current window.
        The initial step is set to 0 if the window is the first one of the experiment.
        If the window is not the first one, the initial step is set to the final step of the previous window.
        """
        if window_id == 0:
            # The first window of the experiment -> start from scratch
            return 0

        prev_window_folder = experiment_path / self._make_window_name(
            window_id - 1,
            window_name_prefix=window_name_prefix,
        )

        if strict:
            assert prev_window_folder.exists(), "Previous window folder does not exist"

        global_step = 0
        if prev_window_folder.exists():
            with open(prev_window_folder / "step_tracker", "r") as f:
                global_step = int(f.read())

        assert global_step >= 0
        return global_step

    @staticmethod
    def get_logging_initial_step_from_checkpoint(
        checkpoint_path: Path,
    ) -> int:
        """
        Returns the global step from the checkpoint file.
        """
        if not checkpoint_path.exists():
            return 0

        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        global_step = loaded_checkpoint["global_step"]

        return global_step

    def make_loggers_resume_parameters(self) -> Dict[str, Any]:
        base_parameters = self.make_loggers_parameters()

        for logger_config, logger_object in zip(
            base_parameters["trainer"]["logger"],
            self.trainer.loggers,
        ):
            if "Wandb" in logger_config["class_path"]:
                exp_id = None
                if getattr(logger_object, "experiment", None) is not None:
                    exp_id = getattr(logger_object.experiment, "id", None)

                logger_config["init_args"]["resume"] = "must"

                if exp_id is None:
                    warnings.warn(
                        "Warning: W&B experiment ID not found. "
                        "The W&B experiment may not be resumed properly. "
                        "Please check your W&B configuration."
                    )
                else:
                    logger_config["init_args"]["id"] = exp_id

        return base_parameters

    def make_loggers_parameters(
        self,
        loggers_config: Optional[Union[Namespace, List[Namespace]]] = None,
        experiment_id: Optional[Union[str, int]] = None,
        window_id: Optional[int] = None,
        experiment_name_prefix: Optional[str] = None,
        window_name_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates the configuration for the loggers to be used in the current window.

        It sets, depending on the logger type, the name and version of the logger to be used.
        The name and version are set to the experiment ID and window ID, respectively.

        For W&B, the group is set to the experiment ID.
        """

        if loggers_config is None:
            loggers_config = self.config[self.subcommand].trainer.logger
        if experiment_id is None:
            experiment_id = self.experiment_id
        if window_id is None:
            window_id = self.window_id
        if experiment_name_prefix is None:
            experiment_name_prefix = self.experiment_info.experiment_name_prefix
        if window_name_prefix is None:
            window_name_prefix = self.experiment_info.window_name_prefix

        experiment_name = f"{experiment_name_prefix}{experiment_id}"
        window_name = f"{window_name_prefix}{window_id}"

        if isinstance(loggers_config, bool):
            # If the logger is a boolean, it means that the user wants to use the default logger
            # (which is TensorBoardLogger by default).
            if loggers_config is False:
                # If the user set the logger to False, we don't need to do anything.
                return {"trainer": {"logger": False}}
            else:
                loggers_config = [
                    jsonargparse.dict_to_namespace(
                        {
                            "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
                            "init_args": {
                                "save_dir": "lightning_logs",
                            },
                        }
                    )
                ]
        elif not isinstance(loggers_config, list):
            loggers_config = [loggers_config]

        result = list()
        logger_conf: Namespace

        for i, logger_conf in enumerate(loggers_config):
            resume_conf = logger_conf.as_dict()

            if "TensorBoard" in logger_conf.class_path:
                resume_conf["init_args"].update(
                    {
                        "name": experiment_name,
                        "version": window_name,
                    }
                )
            elif "Wandb" in logger_conf.class_path:
                resume_conf["init_args"].update(
                    {
                        "name": f"{experiment_name}_{window_name}",
                        "group": experiment_name,
                    }
                )

                if self.offline_logging is not None:
                    resume_conf["init_args"]["offline"] = self.offline_logging
            else:
                raise ValueError(
                    f"Logger {logger_conf} is not supported. Please use TensorBoardLogger or WandbLogger."
                )
            result.append(resume_conf)

        return {"trainer": {"logger": result}}

    def save_experiment_parameters(self, parameters: Optional[Namespace] = None):
        if not self.trainer.strategy.is_global_zero:
            return

        assert self.window_folder is not None

        experiment_parameters_path = self.window_folder / "experiment_parameters.yaml"

        if parameters is None:
            logger_parameters = self.make_loggers_resume_parameters()
            sliding_window_overrides = self._sliding_window_overrides

            assert logger_parameters is not None
            assert sliding_window_overrides is not None

            parameters = Namespace()
            parameters.update(jsonargparse.dict_to_namespace(logger_parameters))
            parameters.update(jsonargparse.dict_to_namespace(sliding_window_overrides))

        self.window_folder.mkdir(parents=True, exist_ok=True)

        with open(experiment_parameters_path, "w") as f:
            yaml.dump(parameters.as_dict(), f)

        print(
            "Saved experiment parameters to",
            str(experiment_parameters_path),
        )
        return experiment_parameters_path

    def find_experiment_parameters_config(self) -> Optional[Path]:
        """
        Returns the path to the experiment parameters file for the current window.
        """
        assert self.window_folder is not None
        params_file = self.window_folder / "experiment_parameters.yaml"
        if not params_file.exists():
            return None
        return params_file

    def make_plugins_parameters(self) -> Dict[str, Any]:
        """
        Creates the configuration for the plugins to be used in the current window.
        """
        assert self.window_folder is not None

        if "callbacks" not in self.config[self.subcommand].trainer:
            # No callbacks defined, no need to add the plugins
            return {"trainer": {"callbacks": dict()}}

        callbacks = self.config[self.subcommand].trainer.callbacks

        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        for i, callback in enumerate(callbacks):
            callback = callback.as_dict()
            if "DeepFakePredictionsDump" in callback.get("class_path", ""):
                if callback["init_args"].get("save_dir", None) is None:
                    callback["init_args"]["save_dir"] = str(self.window_folder)
            callbacks[i] = callback

        return {"trainer": {"callbacks": callbacks}}


def _link_dataset_path(*candidate_paths):
    # Note: --dataset_path takes precedence over dataset_loader.init_args.dataset_path
    first_non_none_path = None
    for candidate_path in candidate_paths:
        if candidate_path is not None and first_non_none_path is None:
            first_non_none_path = candidate_path

        if candidate_path is not None and Path(candidate_path).exists():
            return candidate_path

    return first_non_none_path


__all__ = [
    "LightningCLISlidingWindow",
]
