# CRC-TissueNet - results summary (laptop CPU run)

Generated 2026-11-15 08:07 | profile `capstone`. Hyperparameters were selected on the validation split only; the test cohort (7,180 patches from 50 unseen patients) was used once per final model. Studies used parts of the training data (see each study's settings); validation and test sets were always complete.

## 1. Data

![eda_class_distribution.png](presentation/eda_class_distribution.png)
![eda_example_patches.png](presentation/eda_example_patches.png)
![eda_colour_shift_train_vs_test.png](presentation/eda_colour_shift_train_vs_test.png)

## 2. Baseline CNN - hyperparameter study

| stage | selected setting | val macro F1 |
|---|---|---|
| optimizer | adam | 0.8986 |
| learning_rate | x10 | 0.9482 |
| batch_size | 32 | 0.9492 |
| activation | relu | 0.9492 |
| filter_size | 3x3 | 0.9492 |
| dropout | 0.0 | 0.9588 |

![capstone_cnn_progress.png](presentation/capstone_cnn_progress.png)

![capstone_cnn_stage_optimizer.png](presentation/capstone_cnn_stage_optimizer.png)
![capstone_cnn_stage_learning_rate.png](presentation/capstone_cnn_stage_learning_rate.png)
![capstone_cnn_stage_batch_size.png](presentation/capstone_cnn_stage_batch_size.png)
![capstone_cnn_stage_activation.png](presentation/capstone_cnn_stage_activation.png)
![capstone_cnn_stage_filter_size.png](presentation/capstone_cnn_stage_filter_size.png)
![capstone_cnn_stage_dropout.png](presentation/capstone_cnn_stage_dropout.png)

Full observations: `runs\capstone_cnn/OBSERVATIONS.md`

## 3. Transfer learning - l3_mobilenetv3_small_pt

| stage | selected setting | val macro F1 |
|---|---|---|
| finetune_strategy | fine_tuning | 0.8797 |
| learning_rate | x3 | 0.9172 |

![capstone_transfer_mobilenetv3_small_progress.png](presentation/capstone_transfer_mobilenetv3_small_progress.png)

![capstone_transfer_mobilenetv3_small_stage_finetune_strategy.png](presentation/capstone_transfer_mobilenetv3_small_stage_finetune_strategy.png)
![capstone_transfer_mobilenetv3_small_stage_learning_rate.png](presentation/capstone_transfer_mobilenetv3_small_stage_learning_rate.png)

Full observations: `runs\capstone_transfer_mobilenetv3_small/OBSERVATIONS.md`

## 3. Transfer learning - l4_efficientnet_b0_pt

| stage | selected setting | val macro F1 |
|---|---|---|
| finetune_strategy | fine_tuning | 0.8495 |
| learning_rate | x3 | 0.9121 |

![capstone_transfer_efficientnet_b0_progress.png](presentation/capstone_transfer_efficientnet_b0_progress.png)

![capstone_transfer_efficientnet_b0_stage_finetune_strategy.png](presentation/capstone_transfer_efficientnet_b0_stage_finetune_strategy.png)
![capstone_transfer_efficientnet_b0_stage_learning_rate.png](presentation/capstone_transfer_efficientnet_b0_stage_learning_rate.png)

Full observations: `runs\capstone_transfer_efficientnet_b0/OBSERVATIONS.md`

## 4. Final models on the unseen test cohort

Seeds per model: mean ± SD over seeds; 95% CI = patch-level bootstrap (1000 resamples) for the first seed. Model selection used validation data only.

| ID | Family | Params (M) | Input | Test accuracy | 95% CI | Macro F1 | Macro AUC | Balanced acc | ECE | TUM recall | Train (min) | Infer (ms/img) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN_baseline_default | Scratch | 0.09 | 28 | 0.8593 | 0.852-0.867 | 0.7978 | 0.9634 | 0.7955 | 0.050 | 0.959 | 21.7 | 0.306 |
| CNN_baseline_tuned | Scratch | 0.09 | 28 | 0.9001 | 0.894-0.907 | 0.8651 | 0.9867 | 0.8703 | 0.031 | 0.950 | 153.8 | 4.440 |
| MobileNetV3_small_pretrained | CNN | 1.53 | 64 | 0.9043 | 0.898-0.911 | 0.8738 | 0.9851 | 0.8736 | 0.063 | 0.951 | 12.4 | 0.624 |
| EfficientNet_B0_pretrained | CNN | 4.02 | 64 | 0.9292 | 0.923-0.936 | 0.9029 | 0.9914 | 0.9018 | 0.077 | 0.974 | 49.3 | 2.649 |

Published MedMNIST v2 PathMNIST results (Yang et al. 2023): ResNet-18 (28): acc 0.907, AUC 0.983; ResNet-18 (224): acc 0.909, AUC 0.989; ResNet-50 (28): acc 0.911, AUC 0.99; ResNet-50 (224): acc 0.892, AUC 0.989

**PO1 / H1** - best model by test macro F1: **EfficientNet_B0_pretrained** (accuracy 0.9292, macro AUC 0.9914). Target accuracy ≥ 0.911 and AUC ≥ 0.99: **met** (point estimates; see CIs).

![acc vs params](capstone_benchmark/figures/accuracy_vs_params.png)

![confusion](capstone_benchmark/figures/confusion_best.png) ![reliability](capstone_benchmark/figures/reliability_best.png)

## Per-class test F1 (first seed)

|                              |   ADI |   BACK |   DEB |   LYM |   MUC |   MUS |   NORM |   STR |   TUM |
|:-----------------------------|------:|-------:|------:|------:|------:|------:|-------:|------:|------:|
| CNN_baseline_default         | 0.948 |  0.925 | 0.393 | 0.941 | 0.896 | 0.702 |  0.902 | 0.548 | 0.926 |
| CNN_baseline_tuned           | 0.942 |  0.96  | 0.708 | 0.964 | 0.901 | 0.839 |  0.923 | 0.616 | 0.934 |
| MobileNetV3_small_pretrained | 0.965 |  0.948 | 0.821 | 0.963 | 0.938 | 0.814 |  0.881 | 0.624 | 0.91  |
| EfficientNet_B0_pretrained   | 0.968 |  0.985 | 0.808 | 0.971 | 0.958 | 0.847 |  0.924 | 0.721 | 0.943 |

## Paired comparisons vs CNN_baseline_tuned (same seed, test set)

Paired bootstrap of the macro-F1 difference and McNemar's test; p-values Holm-adjusted.

| id                           | vs                 |   seed |   delta_macro_f1 | delta_ci95         |   p_bootstrap |   p_mcnemar |   only_model_correct |   only_reference_correct |   p_bootstrap_holm |   p_mcnemar_holm | significant_0.05   |
|:-----------------------------|:-------------------|-------:|-----------------:|:-------------------|--------------:|------------:|---------------------:|-------------------------:|-------------------:|-----------------:|:-------------------|
| CNN_baseline_default         | CNN_baseline_tuned |     42 |          -0.0673 | -0.0761 to -0.0578 |         0.001 |      0      |                  194 |                      487 |              0.003 |           0      | True               |
| MobileNetV3_small_pretrained | CNN_baseline_tuned |     42 |           0.0086 | -0.0003 to +0.0180 |         0.068 |      0.2434 |                  324 |                      294 |              0.068 |           0.2434 | False              |
| EfficientNet_B0_pretrained   | CNN_baseline_tuned |     42 |           0.0377 | +0.0291 to +0.0471 |         0.001 |      0      |                  410 |                      201 |              0.003 |           0      | True               |



## 5. Explainability (SHAP)

![shap_CNN_baseline_tuned_channel_importance_gradient-stain.png](presentation/shap_CNN_baseline_tuned_channel_importance_gradient-stain.png)
![shap_CNN_baseline_tuned_channel_importance_gradient.png](presentation/shap_CNN_baseline_tuned_channel_importance_gradient.png)
![shap_CNN_baseline_tuned_gradcam.png](presentation/shap_CNN_baseline_tuned_gradcam.png)
![shap_CNN_baseline_tuned_shap_gradient-stain_correct.png](presentation/shap_CNN_baseline_tuned_shap_gradient-stain_correct.png)
![shap_CNN_baseline_tuned_shap_gradient-stain_errors.png](presentation/shap_CNN_baseline_tuned_shap_gradient-stain_errors.png)
![shap_CNN_baseline_tuned_shap_gradient_correct.png](presentation/shap_CNN_baseline_tuned_shap_gradient_correct.png)
![shap_CNN_baseline_tuned_shap_gradient_errors.png](presentation/shap_CNN_baseline_tuned_shap_gradient_errors.png)
CNN_baseline_tuned: randomisation sanity check, Spearman = 0.293 (low = explanations depend on the learned weights)
![shap_EfficientNet_B0_pretrained_channel_importance_gradient-stain.png](presentation/shap_EfficientNet_B0_pretrained_channel_importance_gradient-stain.png)
![shap_EfficientNet_B0_pretrained_channel_importance_gradient.png](presentation/shap_EfficientNet_B0_pretrained_channel_importance_gradient.png)
![shap_EfficientNet_B0_pretrained_gradcam.png](presentation/shap_EfficientNet_B0_pretrained_gradcam.png)
![shap_EfficientNet_B0_pretrained_shap_gradient-stain_correct.png](presentation/shap_EfficientNet_B0_pretrained_shap_gradient-stain_correct.png)
![shap_EfficientNet_B0_pretrained_shap_gradient-stain_errors.png](presentation/shap_EfficientNet_B0_pretrained_shap_gradient-stain_errors.png)
![shap_EfficientNet_B0_pretrained_shap_gradient_correct.png](presentation/shap_EfficientNet_B0_pretrained_shap_gradient_correct.png)
![shap_EfficientNet_B0_pretrained_shap_gradient_errors.png](presentation/shap_EfficientNet_B0_pretrained_shap_gradient_errors.png)
EfficientNet_B0_pretrained: randomisation sanity check, Spearman = 0.241 (low = explanations depend on the learned weights)
