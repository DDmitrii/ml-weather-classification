# Experiment History

Этот файл хранит журнал экспериментов и ссылки на MLflow runs.

## Planned Experiments

| Config | Run name | Augmentation idea | Status | MLflow page | Best val acc | Test acc | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `configs/experiments/train_exp1_hflip_light.yaml` | `exp1_hflip_light` | horizontal flip | planned | - | - | - | - |
| `configs/experiments/train_exp2_rotation15.yaml` | `exp2_rotation15` | Повороты до 15 градусов | planned | - | - | - | - |
| `configs/experiments/train_exp3_color_jitter.yaml` | `exp3_color_jitter` | Яркость, контраст, насыщенность | planned | - | - | - | - |
| `configs/experiments/train_exp4_flip_rotation.yaml` | `exp4_flip_rotation` | Flip + rotation | planned | - | - | - | - |
| `configs/experiments/train_exp5_flip_color.yaml` | `exp5_flip_color` | Flip + color jitter | planned | - | - | - | - |
| `configs/experiments/train_exp6_strong_aug.yaml` | `exp6_strong_aug` | Более агрессивная смесь аугментаций | planned | - | - | - | - |

## How To Update

После завершения каждого запуска:

1. Найди `summary.json` внутри `mlruns/local_artifacts/experiments/<timestamp>_<run_name>/`.
2. Добавь запись вручную ниже или используй:

```bash
python scripts/update_history.py --summary mlruns/local_artifacts/experiments/<timestamp>_<run_name>/summary.json
```

3. Перенеси ссылку на MLflow run и итоговые метрики в таблицу `Planned Experiments`.

## Experiment Entries

## exp1_hflip_light

- Experiment dir: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light`
- Resolved config: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light\resolved_config.yaml`
- Checkpoint: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light\checkpoints\best_model_state.pth`
- Best epoch: `1`
- Best val accuracy: 0.9851
- Test accuracy: 0.8664
- Test precision: 0.8790
- Test recall: 0.7896
- Test F1: 0.7969
- MLflow experiment: `weather_classification`
- MLflow run id: `d0290604e3294b7bb99d2f9b1b172899`
- MLflow page: http://127.0.0.1:5000/#/experiments/202556990899135500/runs/d0290604e3294b7bb99d2f9b1b172899
- Skipped files report: `-`

### Notes

- Add a short conclusion here after reviewing confusion matrix and class-wise metrics.

## exp1_hflip_light

- Experiment dir: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light`
- Resolved config: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light\resolved_config.yaml`
- Checkpoint: `C:\Users\Karina\PycharmProjects\ml-weather-classification\ml-weather-classification\mlruns\local_artifacts\experiments\20260419_181847_exp1_hflip_light\checkpoints\best_model_state.pth`
- Best epoch: `1`
- Best val accuracy: 0.9851
- Test accuracy: 0.8664
- Test precision: 0.8790
- Test recall: 0.7896
- Test F1: 0.7969
- MLflow experiment: `weather_classification`
- MLflow run id: `d0290604e3294b7bb99d2f9b1b172899`
- MLflow page: http://127.0.0.1:5000/#/experiments/202556990899135500/runs/d0290604e3294b7bb99d2f9b1b172899
- Skipped files report: `-`

### Notes

- Add a short conclusion here after reviewing confusion matrix and class-wise metrics.

