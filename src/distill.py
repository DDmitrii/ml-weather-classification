# src/distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hydra
import omegaconf
import typing
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

# ── Гиперпараметры дистилляции ─────────────────────────────────────
TEMPERATURE   = 4.0    # сглаживание soft labels
ALPHA         = 0.7    # вес KL-loss (soft), (1-alpha) = CE-loss (hard)
EPOCHS        = 40
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
IMG_SIZE      = 224
SAVE_PATH     = "exports/mobilenet_v3_student.pt"


# ── Loss ───────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    def __init__(self, temperature: float, alpha: float):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, C)
        teacher_logits: torch.Tensor,  # (B, C)
        labels: torch.Tensor,          # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard loss: обычная кросс-энтропия
        hard_loss = self.ce(student_logits, labels)

        # Soft loss: KL-дивергенция между сглаженными распределениями
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits   / self.T, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        # Масштабируем T² согласно Hinton et al. (2015)
        soft_loss = kl_loss * (self.T ** 2)

        total = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        return total, hard_loss, soft_loss


# ── Teacher: загружаем обученный ConvNeXt Tiny ─────────────────────
def load_teacher(cfg, device: torch.device) -> nn.Module:
    CKPT_PATH = "checkpoints/epoch=29-val_f1=0.9856.ckpt"

    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # Lightning хранит веса в "state_dict"
    state = checkpoint["state_dict"]

    teacher = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=len(cfg.data.class_names),
    )

    # Lightning добавляет префикс "model." → убираем
    clean_state = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in state.items()
    }

    # Оставляем только ключи самой модели (без head-ов multihead)
    model_keys = set(teacher.state_dict().keys())
    filtered = {k: v for k, v in clean_state.items() if k in model_keys}

    missing, unexpected = teacher.load_state_dict(filtered, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {missing[:5]}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected[:5]}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(device)


# ── Student: MobileNetV3-Small ─────────────────────────────────────
def build_student(num_classes: int) -> nn.Module:
    student = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,       # ImageNet-pretrained backbone
        num_classes=num_classes,
    )
    return student


# ── Eval ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Train loop ─────────────────────────────────────────────────────
@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    from src.data.dataset import WeatherDataset, build_dataloaders
    from src.data.transforms import get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Данные
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Модели
    print("📦 Загружаю teacher (ConvNeXt Tiny)...")
    teacher = load_teacher(cfg, device)

    print("🏗️  Создаю student (MobileNetV3-Small)...")
    student = build_student(len(cfg.data.class_names)).to(device)

    total_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"   Параметры student: {total_params:.1f}M")

    # Оптимизатор + scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    Path("exports").mkdir(exist_ok=True)
    best_acc   = 0.0
    best_epoch = 0

    print(f"\n🚀 Начинаю дистилляцию: {EPOCHS} эпох, T={TEMPERATURE}, α={ALPHA}\n")

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = hard_sum = soft_sum = 0.0
        correct = total = 0

        for batch in train_loader:
            x      = batch[0].to(device)
            labels = batch[1].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_logits = student(x)
                with torch.no_grad():
                    teacher_logits = teacher(x)

                loss, hard, soft = criterion(student_logits, teacher_logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            hard_sum   += hard.item()
            soft_sum   += soft.item()
            correct    += (student_logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()

        train_acc = correct / total
        val_acc   = evaluate(student, val_loader, device)
        lr_now    = scheduler.get_last_lr()[0]
        n_batches = len(train_loader)

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"loss={total_loss/n_batches:.4f} "
            f"(hard={hard_sum/n_batches:.4f}, soft={soft_sum/n_batches:.4f}) | "
            f"train={train_acc:.4f} val={val_acc:.4f} | "
            f"lr={lr_now:.2e}"
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            torch.save({
                "epoch":      epoch,
                "val_acc":    val_acc,
                "state_dict": student.state_dict(),
                "cfg":        cfg,
            }, SAVE_PATH)
            print(f"   💾 Сохранён лучший checkpoint (val_acc={val_acc:.4f})")

    print(f"\n✅ Дистилляция завершена!")
    print(f"   Лучшая val accuracy : {best_acc:.4f} (epoch {best_epoch})")
    print(f"   Checkpoint          : {SAVE_PATH}")
    print(f"\n📊 Сравнение:")
    print(f"   Teacher (ConvNeXt Tiny)   : 99.08% acc, ~106 MB fp32")
    print(f"   Student (MobileNetV3-Small): {best_acc:.2%} acc")


if __name__ == "__main__":
    main()