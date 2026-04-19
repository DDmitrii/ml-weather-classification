import hashlib
import os
import shutil
import sys
import zipfile

import requests
from tqdm import tqdm

YANDEX_PUBLIC_URL = "https://disk.yandex.ru/d/kUr2EqYlocfLjw"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
SPLITS = ("train", "val", "test")


def get_direct_url(public_url: str) -> str:
    """Получить прямую ссылку через Яндекс.Диск API."""
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    response = requests.get(api, params={"public_key": public_url})
    response.raise_for_status()
    return response.json()["href"]


def download_file(url: str, dest: str) -> None:
    """Скачать файл с прогресс-баром."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc="📦 Скачиваю датасет",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def sanitize_filename(filename: str) -> str:
    """
    Если имя файла слишком длинное — заменяем на md5-хэш.
    Решает OSError: [Errno 36] File name too long.
    """
    if len(filename.encode("utf-8")) > 200:
        ext = os.path.splitext(filename)[1].lower()
        return hashlib.md5(filename.encode()).hexdigest()[:12] + ext
    return filename


def unzip(zip_path: str, dest_dir: str) -> None:
    """
    Распаковать архив в dest_dir.
    - Убирает корневую папку архива (weather_dataset/, __MACOSX/ и т.п.)
    - Переименовывает файлы с длинными именами
    """
    print("📂 Распаковываю архив...")
    renamed_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        for member in tqdm(members, desc="   Извлекаю", unit="файл"):
            parts = member.filename.split("/")

            # Пропускаем служебные папки macOS
            if parts[0] in ("__MACOSX",):
                continue

            # Убираем корневую папку архива если есть
            # (папка считается корневой если не совпадает с train/val/test)
            if len(parts) > 1 and parts[0] not in SPLITS:
                parts = parts[1:]

            if not parts or parts[0] == "":
                continue

            # Чиним длинные имена файлов
            parts[-1] = sanitize_filename(parts[-1])
            if parts[-1] != member.filename.split("/")[-1]:
                renamed_count += 1

            target_path = os.path.join(dest_dir, *parts)

            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    os.remove(zip_path)
    print(f"🗑️  Архив удалён")
    if renamed_count:
        print(f"✏️  Переименовано файлов с длинными именами: {renamed_count}")


def print_stats() -> None:
    """Вывести статистику по датасету."""
    print("\n✅ Готово! Структура датасета:")
    for split in SPLITS:
        split_path = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_path):
            print(f"  {split:5s}: ⚠️  папка не найдена")
            continue
        classes = sorted(
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        )
        total = sum(
            len(os.listdir(os.path.join(split_path, c))) for c in classes
        )
        print(f"  {split:5s}: {total:6d} изображений | {len(classes)} классов: {classes}")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # Если данные уже есть — пропускаем
    if all(os.path.isdir(os.path.join(DATA_DIR, s)) for s in SPLITS):
        print("✅ Данные уже скачаны. Пропускаю.")
        print_stats()
        sys.exit(0)

    print("🔗 Получаю прямую ссылку с Яндекс.Диска...")
    direct_url = get_direct_url(YANDEX_PUBLIC_URL)

    download_file(direct_url, ZIP_PATH)
    unzip(ZIP_PATH, DATA_DIR)
    print_stats()


if __name__ == "__main__":
    main()