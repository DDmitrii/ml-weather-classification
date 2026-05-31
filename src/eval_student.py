import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import timm
import typing
import omegaconf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import hydra

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

STUDENT_CKPT = "exports/mobilenet_v3_student.pt"
TEACHER_CKPT = "checkpoints/epoch=17-val_f1=0.9921.ckpt"


@torch.no_grad()
def evaluate(model, loader, device, use_multihead=False) -> tuple[float, dict]:
    from collections import defaultdict

    model.eval()
    correct = total = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)

        if use_multihead:
            logits_dn, logits_wt = model(x)
            preds = model._combine_preds(logits_dn, logits_wt)
        else:
            preds = model(x).argmax(1)

        correct += (preds == y).sum().item()
        total   += y.size(0)
        for pred, label in zip(preds.cpu(), y.cpu()):
            per_class_total[label.item()]   += 1
            per_class_correct[label.item()] += int(pred == label)

    per_class_acc = {
        k: per_class_correct[k] / per_class_total[k]
        for k in sorted(per_class_total)
    }
    return correct / total, per_class_acc


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    from src.data import WeatherDataset, get_val_transforms, build_dataloaders
    from src.model import WeatherClassifierMultiHead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(cfg.data.class_names)

    # class_weights — как в test_only.py
    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=class_names,
        transform=get_val_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()

    test_ds = WeatherDataset(
        root=cfg.data.test_dir,
        class_names=class_names,
        transform=get_val_transforms(cfg.data.img_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0
    )
    print(f"📂 Test set: {len(test_ds)} изображений\n")

    # ── Teacher ───────────────────────────────────────────────────────────────
    teacher = WeatherClassifierMultiHead.load_from_checkpoint(
        TEACHER_CKPT,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )
    teacher = teacher.to(device)

    # ── Student ───────────────────────────────────────────────────────────────
    student = timm.create_model(
        "mobilenetv3_small_100", pretrained=False, num_classes=len(class_names)
    )
    ckpt = torch.load(STUDENT_CKPT, map_location=device, weights_only=False)
    sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    student.load_state_dict(sd)
    student = student.to(device)

    # ── Замер ─────────────────────────────────────────────────────────────────
    print("Считаю метрики teacher (MultiHead)...")
    teacher_acc, teacher_per = evaluate(teacher, test_loader, device, use_multihead=True)

    print("Считаю метрики student...")
    student_acc, student_per = evaluate(student, test_loader, device, use_multihead=False)

    # ── Таблица ───────────────────────────────────────────────────────────────
    print(f"\n{'Класс':<20} {'Teacher':>10} {'Student':>10} {'Разница':>10}")
    print("─" * 55)
    for i, name in enumerate(class_names):
        t    = teacher_per[i]
        s    = student_per[i]
        diff = s - t
        flag = "⚠️  " if diff < -0.05 else "   "
        print(f"{flag}{name:<20} {t:>9.2%} {s:>9.2%} {diff:>+9.2%}")

    print("─" * 55)
    print(f"{'   ИТОГО':<20} {teacher_acc:>9.2%} {student_acc:>9.2%} "
          f"{student_acc - teacher_acc:>+9.2%}")


if __name__ == "__main__":
    main()