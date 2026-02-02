# Training and evaluation of a detector
We provide a Lightning-based framework to train a detector on the AI-GenBench.

The proposed AI-GenBench benchmark requires a detector to be trained on sliding windows of 4 generators, ordered chronologically. For more info, please refer to our paper [AI-GenBench: A New Ongoing Benchmark for AI-Generated Image Detection](https://arxiv.org/abs/2504.20865).
This framework can also be used to train on the dataset without following the benchmark protocol.

## Update
The latest version also brings the code needed to reproduce the experiments found in our paper [Generalized Design Choices for Deepfake Detectors](https://arxiv.org/abs/2504.20865). The following guide applies to those as well! For the specific configurations, please check [the paper-specific readme](training_configurations/generalized_choices_journal/README.md).

## Getting started

1. Clone the [repository](../).
2. Follow the guide to install training/evaluation requirements (see the [README in the root of the repository](../README.md)) in a Python environment
3. Follow the [Training](#training) section below

## Training
You can either follow the benchmark protocol or train a model directly on the whole dataset. The following sections describe both options.

### Benchmark protocol

1. Create a `local_config.yaml` file. This file will contain the local configuration parameters, such as the dataset path (but you can also set other configuration values). The file should look like this:
    ```
    trainer:
        precision: "bf16-mixed"
    dataset_path: "/folder/subfolder/.../ai_gen_bench_v1.0.0"
    ```
2. (Optional) Adapt the parameters for one of the already provided models. The smallest model we evaluated in [our paper](https://arxiv.org/abs/2504.20865) is OpenAI ResNet50 CLIP, whose configuration is in [`RN50_clip_tune_resize.yaml`](training_configurations/RN50_clip/RN50_clip_tune_resize.yaml). The configuration files for other models are similar.
3. Run the training script. We recommend using `run_training.sh`, but you may also do it manually:
    ```bash
    python lightning_main.py fit \
    --config training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml \
    --config training_configurations/RN50_clip/RN50_clip_tune_resize.yaml \
    --config local_config.yaml \
    --experiment_info.experiment_id <experiment_id>
    ```
    Just make sure that the `benchmark_pipelines` config is before the model config.
4. Logs will be stored in the `experiments_logs` folder, while predictions and checkpoints will be stored in the `experiments_data` folder.
    - By default, predictions will be saved at the end of the "validate" and "test" phases as a `.npz` file, which can be easily loaded using `NumPy`.
    - The framework will create multiple folders in `experiments_data` to store the results of each sliding window. Each will have its own checkpoint and predictions file.
    - When using W&B Logger, each experiment will have its own "group" set, so you can easily group the sliding windows runs in the online dashboard.
    - When using TensoboardLogger, logs will be stored in `experiments_logs`. For each experiment, each window will have its own `window_N` folder. Hint: you can aggregate all tfevent files in a single folder to visualize them as a single experiment :).
    - It is recommended you set the `--experiment_info.experiment_id` parameter to a meaningful value from the command line, so that you can easily identify the experiment in the logs. By default, a incremental id will be set (it's the default LightningCLI behavior).

### Training on the whole dataset
If you just want to train a detector on the whole dataset without following the proposed benchmark protocol, you can do so by following the steps detailed in [Benchmark protocol](#benchmark-protocol) with the following differences:

1. Set the dataset path in [`base_benchmark_full_training.yaml`](training_configurations/benchmark_pipelines/base_benchmark_full_training.yaml).
2. Run the training script as described above.

### Resuming a training
There are situations in which you may want to pause and then resume a training. The framework should already take care of SLURM preemption and re-queue if enabled (the experiment id will be automatically detected as the job id), but also resuming an experiment stopped using external signals (CTRL-C) should work.

In the last case, you can resume a training by running the same command you used to start the training, but adding the experiment id:
```bash
python lightning_main.py fit \
    --config training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml \
    --config training_configurations/RN50_clip/RN50_clip_tune_resize.yaml \
    --config local_config.yaml \
    --experiment_info.experiment_id <experiment_id>
```

## Customizing the training procedure
You can customize the training procedure mainly by modifying the model configuration files such as [`RN50_clip_tune_resize.yaml`](training_configurations/RN50_clip/RN50_clip_tune_resize.yaml). Those are the most relevant configuration values to consider:
- `model.model_name`: must be set to a value recognized by the model factory (see below for more details). All models come in the `_probe` and `_tune` variants. The `_probe` variant is used to train the only the last layer of the model, while the `_tune` variant is used to train the full model.
- `optimizer`: you can set any optimizer you want. Set the `class_path` to the optimizer you want to use and its hyperparameters in `init_args`.
- `scheduler`: you will need to implement schedulers manually in the code. Here we already implemented `OneCycleLR`, which is configured as a string and actually implemented in the `configure_optimizers` method of [the model](algorithms/base_model.py).
- `model_input_size`: defaults to `224` as most models will work with 224x224 images, but you can customize the input size here.
- `classification_threshold`: only affects some metrics (accuracy, precision, recall). AUROC is not affected. Doesn't affect the training process.
- `training_cropping_strategy` / `evaluation_cropping_strategy`: you can set any cropping strategy you want. In our paper we found that the `resize` strategy works generally better. You will find configuration files for both `resize` and `crop` in the [`training_configurations`](training_configurations/) folder. Valid values are:
    - training_cropping_strategy: `resize`, `random_crop`, `center_crop`, `as_is`
    - evaluation_cropping_strategy: `resize`, `crop` (central), `multicrop` (implemented as [FiveCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.FiveCrop.html) by default, can be customized in the codebase), `as_is`
- Keep an eye to `trainer.accumulate_grad_batches` and `data.train_batch_size`: those are important parameters to set the batch size. The `trainer.accumulate_grad_batches` parameter is used to set the number of batches to accumulate before performing an update pass. This is useful when you want to simulate a larger batch size than what your GPU can handle. The `data.train_batch_size` parameter is used to set the batch size for the training data loader. The overall batch size is the product of those two parameters. Also, if you are using multiple GPUs, the overall batch size will be multiplied by the number of GPUs. So, if you are using 4 GPUs and set `trainer.accumulate_grad_batches` to 2 and `data.train_batch_size` to 32, the overall batch size will be 4 * 2 * 32 = 256. We ran all paper experiments with an overall batch size of 512.

## Adding your model
IMPORTANT: here we refer to the process of registering a model architecture such as a new ResNet, ViT, EfficientNet, etcetera, in the model factory. This is not the same as customizing the provided LightningModule [`BaseDeepfakeDetectionModel`](algorithms/base_model.py). That advanced step is discussed later.

You don't need to implement a new LightningModule class (LightningModule is the superclass used in Lightning to define the training and evaluation logic) to add a new model . Instead, you can follow these steps (you can check [dinov2.py](algorithms/models/dinov2.py) for a general template):

1. In `algorithms/models`, create a new file called `your_model.py`.
2. In `your_model.py`, implement the model factory, such as:
    ```python
    def make_dinov2_model(model_name: str, pretrained: bool = True, **kwargs):
        ...
    ```
    The factory should return, for recognized model names, a PyTorch model or None/raise an error if the model name is not recognized.
3. Register your model factory (the first parameter is just a name you want to use to refer to the factory):
    ```python
    from algorithms.model_factory_registry import ModelFactoryRegistry

    ModelFactoryRegistry().register_model_factory("dinov2", make_dinov2_model)
    ```
4. Import your script somewhere in the codebase. For example, in the `__init__.py` file of the `algorithms/models` folder. This will ensure that your model factory is registered when the package is imported.
5. That's it! From now on you can use your model by setting the `model_name` in the config files.
6. (Advanced) For all the pre-implemented models (RN50_CLIP, DINOv2, OpenViT-L-14) we created two versions: `probe` and `tune`. In `probe`, we "hid" the backbone so that it's not listed in the sub-modules and parameters (and thus is not accidentaly taken into consideration during training). You might not need to implement a probe-only version of your model, but if you need it you follow the general schema found in [dinov2.py](algorithms/models/dinov2.py).
    - Setting `requires_grad` to `False` is not enough to freeze a part of the model: you'll also need to `eval()` the modules to be frozen of otherwise the stats of normalization layers will change (they are not frozen by setting `requires_grad=False`). Also, for frozen parts of the model, it would be better to forward inputs using `torch.no_grad()` or `torch.inference_mode()` to avoid unnecessary memory usage and computation. This is not strictly necessary, but it's a good practice.

### Already implemented models
- [DINOv2 models](https://github.com/facebookresearch/dinov2), either `_probe` or `_tune`. Example: `dinov2_vits14_tune`.
- [OpenAI CLIP models](https://github.com/openai/CLIP) models, either `_probe` or `_tune`. Example: `RN50_tune`.
- A subset of [OpenCLIP models](https://github.com/mlfoundations/open_clip), either `_probe` or `_tune`. See [openclip_models.py](algorithms/models/openclip_models.py) for the supported ones. Example: `clipL14commonpool_tune`.
- [timm](https://huggingface.co/docs/hub/timm) models, either `_probe` or `_tune`. Example: `convnext_xxlarge.clip_laion2b_soup_ft_in1k_tune`
- [torchvision](https://pytorch.org/vision/main/models.html) models, either `_probe` or `_tune`. Example: `resnet50_tune`

## Customizing the training and evaluation logic or augmentations
This is an advanced step. You can customize the training and evaluation logic in the [`BaseDeepfakeDetectionModel`](algorithms/base_model.py) by either modifying, subclassing, or even just using it as a template. The class is already documented in-code to guide you through the process.

The main things you may want to change are:
- `training_step()`: the default implementation already takes care of computing the loss and updating metrics. The only strict requirement is that it must return the loss.
- `evaluation_step()`: the default implementation already takes care of computing the loss, fusing the scores if using multi_crop evaluation, and updating metrics. Note: `evaluation_step` is a unified method for both validation and test. The only strict requirement is that it should return (predictions, labels, generator_ids, image_identifiers, losses).
- `scores_fusion()`: only meaningful when running a multicrop evaluation. The default implementation computes the mean score over the crops.
- `configure_optimizers()`: the default implementation already takes care of setting the optimizer and the OneCycleLR scheduler.
    - The optimizer_factory is a factory (created by Lightning from the YAML configuration) that accepts the parameter groups and returns the optimizer.
    - You will need to manually implement the scheduler. Follow the implementation of OneCycleLR as a general template. Note that in Lightning it's not exactly straightforward to implement a scheduler, so you may need to play around a bit.
    - Consider customizing `lr_scheduler_step()` only if you really need a custom scheduler step function.
- `register_metrics()`: you can override this method to add new metrics. Just remember to call `super().register_metrics()` to configure the default metrics! Check the `register_metrics` implementation in the superclass to see how to add new metrics. You can use any metric from the `torchmetrics` library, but you can also implement your own metrics.
  - We are planning to make this process more intuitive in the future.
- `train_augmentation()`: must return 1 or 2 augmentations (callables):
    1. The first one will be the deterministic part of the pipeline (actually, it will be deterministic only if `deterministic_augmentations` is `true` in the configuration)
    2. The second, optional, one is the stochastic part of the pipeline. If you want to follow the benchmark protocol, only limited agumentations are allowed to be in this part: flip, rotation, crop.
- `val/test/predict_augmentation()`: must return 2 augmentations (callables). Both will be executed in a deterministic context if `deterministic_augmentations` is `true` in the configuration. To understand this, consider that this is how the augmentation pipeline looks like:
    ```python
    first_augmentation, second_augmentation = model.val_augmentation()
    # mandatory augmentations are defined in the benchmark pipeline class
    img = pipeline.mandatory_eval_processing(img)        # 1
    img = first_augmentation(img)                        # 2
    # make_val_crops is a method of the model, explained below
    crops =  model.make_val_crops(img)
    result_crops = []
    for crop in crops:
        result_crops.append(second_augmentation(crop))   # 3
    return result_crops
    ```
    1. Firstly, the mandatory augmentations (defined by the benchmark pipeline) are executed (see below).
    2. The first augmentation is applied before the "crop/multicrop/resize" part of the pipeline.
    3. The second augmentation is applied separatedly to each crop.
- `make_eval_crops()`: this method is used to create the crops (for evaluation only). The default implementation already takes care of resizeing or cropping or multi-cropping the image based on the `evaluation_cropping_strategy` configuration value. You can override it if you want to change the cropping strategy.

That's it! Just remember that, in [ai_genbench_pipeline.py](lightning_data_modules/ai_genbench_pipeline.py), **a part of the evaluation augmentations are mandatory (only if you want to follow the benchmark protocol, of course)**. The **mandatory augmentations** are designed to create images that **could be realistically considered as "real"** and thus used on social media posts/newspapers/..., while still adding distorsion, compressions, etcetera. You can find the mandatory augmentations in the `mandatory_val_preprocessing()` method of the `AIGenBenchPipeline` class. You can still add your own augmentations in the model's `val/test/predict_augmentation()` methods, that will be executed *after* the mandatory ones (mandatory ones are used to mimic a situation in which the detection system receives images as they were already distorted using that pipeline).
