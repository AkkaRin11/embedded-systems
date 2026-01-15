import cv2
import os
import shutil
import random
from ultralytics import YOLO


def load_yolo_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    print(f"Модель загружена: {model_path}")
    return model


def detect_cups_only(model, image_path, conf_threshold=0.8):
    # В Ultralytics 8.x классы передаются списком. Класс 41 - это cup в COCO.
    results = model(image_path, conf=conf_threshold, classes=[41], verbose=False)
    return results[0]


def create_output_folders(base_path="annotated_dataset"):
    folders = [
        os.path.join(base_path, "train", "images"),
        os.path.join(base_path, "train", "labels"),
        os.path.join(base_path, "val", "images"),
        os.path.join(base_path, "val", "labels"),
        os.path.join(base_path, "visualizations")
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print(f"Созданы папки в: {base_path}")
    return base_path


def save_yolo_annotation(boxes, image_width, image_height, output_path):
    with open(output_path, 'w') as f:
        for box in boxes:
            # Преобразуем тензор в список для итерации (совместимость с PyTorch 2.4.1)
            coords = box.xywhn[0].tolist()
            x_center, y_center, width, height = coords
            class_id = 0  # Мы размечаем только один класс - стаканы
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def is_quality_detection(boxes, min_confidence=0.8):
    if len(boxes) == 0:
        return False

    for box in boxes:
        # Извлечение значения конфиденса как float
        conf_val = float(box.conf[0].item())
        if conf_val < min_confidence:
            return False

    return True


def process_dataset(input_folder, output_folder, model, train_split=0.8, visualize=True, min_conf=0.8):
    print(f"\n=== ОБРАБОТКА ДАТАСЕТА ===")
    print(f"Входная папка: {input_folder}")
    print(f"Класс: cup (41 в COCO)")
    print(f"Минимальная уверенность: {min_conf * 100:.0f}%")
    print(f"Train/Val split: {train_split * 100:.0f}% / {(1 - train_split) * 100:.0f}%\n")

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    quality_images = []

    print("Фильтрация изображений по качеству детекции...")
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(input_folder, filename)

        results = detect_cups_only(model, image_path, conf_threshold=min_conf)
        boxes = results.boxes

        if len(boxes) > 0 and is_quality_detection(boxes, min_conf):
            avg_conf = sum(float(box.conf[0].item()) for box in boxes) / len(boxes)
            quality_images.append((filename, len(boxes), avg_conf))
            print(f"  {idx + 1}/{len(image_files)}: {filename} - {len(boxes)} стаканов (conf: {avg_conf:.2f})")
        else:
            print(f"  {idx + 1}/{len(image_files)}: {filename} - отклонено (низкая уверенность или нет стаканов)")

    if not quality_images:
        print(f"\nНет изображений с уверенностью > {min_conf*100}%!")
        return None

    random.shuffle(quality_images)

    split_idx = int(len(quality_images) * train_split)
    train_files = [item[0] for item in quality_images[:split_idx]]
    val_files = [item[0] for item in quality_images[split_idx:]]

    print(f"\n=== ОТОБРАННЫЕ ИЗОБРАЖЕНИЯ ===")
    print(f"Всего качественных: {len(quality_images)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}\n")

    stats = {
        'train_images': 0,
        'train_cups': 0,
        'val_images': 0,
        'val_cups': 0,
        'total_cups': 0
    }

    for split_name, files in [('train', train_files), ('val', val_files)]:
        print(f"Сохранение {split_name}...")

        for idx, filename in enumerate(files):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Ошибка загрузки: {filename}")
                continue

            height, width = image.shape[:2]

            results = detect_cups_only(model, image_path, conf_threshold=min_conf)
            boxes = results.boxes

            image_output = os.path.join(output_folder, split_name, "images", filename)
            label_output = os.path.join(output_folder, split_name, "labels", f"{os.path.splitext(filename)[0]}.txt")

            shutil.copy(image_path, image_output)
            save_yolo_annotation(boxes, width, height, label_output)

            stats[f'{split_name}_images'] += 1
            stats[f'{split_name}_cups'] += len(boxes)
            stats['total_cups'] += len(boxes)

            if visualize:
                annotated = results.plot()
                vis_output = os.path.join(output_folder, "visualizations", f"{split_name}_{filename}")
                cv2.imwrite(vis_output, annotated)

            avg_conf = sum(float(box.conf[0].item()) for box in boxes) / len(boxes)
            print(f"  {idx + 1}/{len(files)}: {filename} - {len(boxes)} стаканов (conf: {avg_conf:.2f})")

    print(f"\n=== ФИНАЛЬНАЯ СТАТИСТИКА ===")
    print(f"Train: {stats['train_images']} изображений, {stats['train_cups']} стаканов")
    print(f"Val: {stats['val_images']} изображений, {stats['val_cups']} стаканов")
    print(f"Всего стаканов: {stats['total_cups']}")

    return stats


def create_yaml_file(output_folder, class_name="cup"):
    yaml_content = f"""path: {os.path.abspath(output_folder)}
train: train/images
val: val/images

nc: 1
names: ['{class_name}']
"""

    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nСоздан файл: {yaml_path}")
    return yaml_path


def main():
    print("=== АВТОМАТИЧЕСКАЯ РАЗМЕТКА СТАКАНОВ (ВЫСОКОЕ КАЧЕСТВО) ===\n")

    # В Python 3.8 input() возвращает str, что нам и нужно
    input_folder = input("Путь к папке с изображениями (например, dataset_20251209_173000/color): ").strip()

    if not os.path.exists(input_folder):
        print(f"Папка не найдена: {input_folder}")
        return

    output_folder = "annotated_cups_dataset"
    output_folder = create_output_folders(output_folder)

    # Используем базовую модель для разметки
    model = load_yolo_model("yolov8n.pt")

    train_split = 0.8
    min_confidence = 0.8

    stats = process_dataset(input_folder, output_folder, model, train_split, visualize=True, min_conf=min_confidence)

    if stats is None:
        return

    yaml_path = create_yaml_file(output_folder)

    print("\n=== ГОТОВО! ===")
    print(f"Датасет: {output_folder}")
    print(f"Конфиг: {yaml_path}")
    print(f"\nДля обучения YOLOv8:")
    print(f"yolo train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == '__main__':
    main()
