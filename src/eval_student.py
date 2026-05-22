# src/eval_student.py
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
TEACHER_CKPT = "checkpoints/epoch=29-val_f1=0.9856.ckpt"


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict]:
    from collections import defaultdict
    import torch.nn.functional as F

    model.eval()
    correct = total = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        preds  = logits.argmax(1)
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
    from src.data.dataset import WeatherDataset
    from src.data.transforms import get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(cfg.data.class_names)

    test_ds = WeatherDataset(
        root=cfg.data.test_dir,
        class_names=class_names,
        transform=get_val_transforms(cfg.data.img_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0
    )
    print(f"📂 Test set: {len(test_ds)} изображений\n")

    # ── Teacher ───────────────────────────────────────────────────
    teacher = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(class_names))
    ckpt    = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    state   = ckpt["state_dict"]
    clean   = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state.items()}
    model_keys = set(teacher.state_dict().keys())
    teacher.load_state_dict({k: v for k, v in clean.items() if k in model_keys}, strict=False)
    teacher = teacher.to(device)

    teacher_acc, teacher_per_class = evaluate(teacher, test_loader, device)

    # ── Student ───────────────────────────────────────────────────
    student = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(class_names))
    ckpt    = torch.load(STUDENT_CKPT, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["state_dict"])
    student = student.to(device)

    student_acc, student_per_class = evaluate(student, test_loader, device)

    # ── Отчёт ─────────────────────────────────────────────────────
    print(f"{'Класс':<20} {'Teacher':>10} {'Student':>10} {'Разница':>10}")
    print("─" * 55)
    for i, name in enumerate(class_names):
        t = teacher_per_class[i]
        s = student_per_class[i]
        diff = s - t
        flag = "⚠️ " if diff < -0.05 else ""
        print(f"{flag}{name:<20} {t:>9.2%} {s:>9.2%} {diff:>+9.2%}")

    print("─" * 55)
    print(f"{'ИТОГО':<20} {teacher_acc:>9.2%} {student_acc:>9.2%} {student_acc - teacher_acc:>+9.2%}")


if __name__ == "__main__":
    main()