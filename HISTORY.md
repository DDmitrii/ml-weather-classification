# Журнал экспериментов

Этот файл хранит журнал экспериментов и ссылки на MLflow runs

## Эксперементы

| Config | Run name | Augmentation idea | Status | MLflow page | Best val acc | Test acc | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/experiments/train_exp1_hflip_light.yaml` | `exp1_hflip_light` | horizontal flip | planned | - | - | - | - |
| `configs/experiments/train_exp2_rotation15.yaml` | `exp2_rotation15` | Повороты до 15 градусов | planned | - | - | - | - |
| `configs/experiments/train_exp3_color_jitter.yaml` | `exp3_color_jitter` | Яркость, контраст, насыщенность | planned | - | - | - | - |
| `configs/experiments/train_exp4_flip_rotation.yaml` | `exp4_flip_rotation` | Flip + rotation | planned | - | - | - | - |
| `configs/experiments/train_exp5_flip_color.yaml` | `exp5_flip_color` | Flip + color jitter | planned | - | - | - | - |
| `configs/experiments/train_exp6_strong_aug.yaml` | `exp6_strong_aug` | Более агрессивная смесь аугментаций | planned | - | - | - | - |

## exp1_hflip_light

- Experiment dir: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_223346_exp1_hflip_light`
- Resolved config: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_223346_exp1_hflip_light\resolved_config.yaml`
- Checkpoint: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_223346_exp1_hflip_light\checkpoints\best_model_state.pth`
- Best epoch: `1`
- Best val accuracy: 0.9869
- Test accuracy: 0.8693
- Test precision: 0.8812
- Test recall: 0.7926
- Test F1: 0.7999
- MLflow experiment: `weather_classification`
- MLflow run id: `245591f20b3c493889e477b9f4b6ab40`
- MLflow page: http://127.0.0.1:5000/#/experiments/202556990899135500/runs/245591f20b3c493889e477b9f4b6ab40

## exp7_convnext_tiny_randaug_ls

- Experiment dir: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260420_011729_exp7_convnext_tiny_randaug_ls`
- Resolved config: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260420_011729_exp7_convnext_tiny_randaug_ls\resolved_config.yaml`
- Checkpoint: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260420_011729_exp7_convnext_tiny_randaug_ls\checkpoints\best_model_state.pth`
- Best epoch: `9`
- Best val accuracy: 0.9879
- Test accuracy: 0.8860
- Test precision: 0.9088
- Test recall: 0.7991
- Test F1: 0.8053
- MLflow experiment: `weather_classification`
- MLflow run id: `eb31b9563b20415796a2d6f00daf9408`
- MLflow page: http://127.0.0.1:5000/#/experiments/202556990899135500/runs/eb31b9563b20415796a2d6f00daf9408

