# Generalized Design Choices for Deepfake Detectors

## Content
This folder contains the configuration files of the experiments of our paper [**Generalized Design Choices for Deepfake Detectors**](https://arxiv.org/abs/2511.21507).

The configurations are divided into model-specific directories. To run the experiments, follow steps recommended in the [README in the root project](../../../README.md): you'll need to setup the [AI-GenBench dataset](../../../dataset_creation/README.md) and follow the guide to [run the training code](../../README.md).

In particular, you'll need to pass 3 configuration files (+ local config, optional):

1. The benchmark configuration, usually `training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml`
2. The general model training configration, such as `training_configurations/generalized_choices_journal/DINOv2/DINOv2_tune_resize_generic.yaml`
3. The experiment-specific customization configuration, such as `training_configurations/generalized_choices_journal/DINOv2/DINOv2_tune_resize_am14.yaml`

For instance, this can be used to run the harmonic replay experiment on DINOv2:

```bash
python lightning_main.py fit \
--config training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml \
--config training_configurations/generalized_choices_journal/DINOv2/DINOv2_tune_resize_generic.yaml \
--config training_configurations/generalized_choices_journal/DINOv2/DINOv2_tune_resize_am4_continual_harmonic_unbounded.yaml \
--config local_config.yaml
```

Note: the experiment id is usually hard-coded in the experiment-specific configuration.

## Overview of the configuration files

- `<model-name>_tune_resize_amN.yaml`: used in conjunction with `DINOv3_tune_resize_generic.yaml`, to reproduce the *am* variation experiments.
    - In particular `..._am4.yaml` + `..._resize_generic.yaml` is what we refer to as the *baseline*.
- `<model-name>_tune_resize_amN.yaml`: used in conjunction with `..._resize_generic.yaml`, to reproduce the *am* variation experiments (Table 2 and Figure 3).
- `...soft_train_aug.yaml_am4` + `...soft_train_aug.yaml`: to use the *Evaluation* augmentation pipeline
    - Used in Table 1 and Figure 2
- `...mild_train_aug.yaml_am4` + `...mild_train_aug.yaml`: to use the *Mild* augmentation pipeline
    - Used in Table 1 and Figure 2
- `..._resize_am4_epochsN.yaml` + `..._resize_generic.yaml`: to reproduce the epochs variation experiments (Table 3 and Figure 3)
- `...resize/crop_eval_mix_am4.yaml` + `...resize/crop_eval_mix.yaml`: to reproduce input processing experiments (Table 4 and Figure 4)
- `..._resize_am4_multiclass.yaml` + `..._resize_generic.yaml`: to reproduce the plain multiclass training (Table 5 and Figure 5)
    - Note: scores are fused using baseline, max, and sum using scripts found in [`evaluation_scripts`](../../evaluation_scripts/)
- `...double_head_(binary_atop|separate).yaml` + specialization `(am4 / am4_bin0.75)`: used to reproduce dual head experiments (Table 6 and Figure 6)
- `<model-name>_tune_resize_knn.yaml` + `..._knn_k1_cN_am4.yaml`: used for distance-based scoring experiments (by varying C=number of centroids). Used in Table 7 and Figure 7.
- `..._resize_generic.yaml` + `...reset_weights/continual_harmonic_unbounded/continual_cb/continual_cb_20k.yaml`: used to reproduce continual learning experiments (Table 8, Table 9, Figure 8, Figure 9).
