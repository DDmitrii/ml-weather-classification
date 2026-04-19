from pathlib import Path
from collections import Counter
from PIL import Image
import os

SPLITS = ("train", "val", "test")


def check_dataset(data_dir: str = "src/data/data") -> None:
    """Проверить структуру датасета и вывести статистику."""
    data_path = Path(data_dir)
    print("=" * 50)
    print("📊 Статистика датасета")
    print("=" * 50)

    all_classes = None

    for split in SPLITS:
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ {split}: папка не найдена")
            continue

        classes = sorted(d.name for d in split_path.iterdir() if d.is_dir())
        counts = {}
        broken = []

        for cls in classes:
            cls_path = split_path / cls
            imgs = list(cls_path.glob("*"))
            imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            counts[cls] = len(imgs)

            # Проверяем что файлы открываются
            for img_path in imgs[:3]:
                try:
                    Image.open(img_path).verify()
                except Exception:
                    broken.append(str(img_path))

        total = sum(counts.values())
        print(f"\n📁 {split.upper()} — {total} изображений")
        for cls, cnt in counts.items():
            bar = "█" * (cnt // max(counts.values()) * 20 // 1) if counts else ""
            print(f"   {cls:<20} {cnt:>5}")

        if broken:
            print(f"   ⚠️  Битых файлов: {len(broken)}")

        if all_classes is None:
            all_classes = classes
        elif classes != all_classes:
            print(f"   ⚠️  Классы в {split} не совпадают с train!")

    print("\n" + "=" * 50)
    if all_classes:
        print(f"✅ Классов: {len(all_classes)}")
        print(f"   {all_classes}")
    print("=" * 50)


if __name__ == "__main__":
    check_dataset()