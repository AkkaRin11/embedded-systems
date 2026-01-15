import cv2
import os
import torch
import numpy as np
from datetime import datetime
import time
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("Внимание: библиотека pyrealsense2 не установлена. Работа с камерой RealSense будет недоступна.")


def suppress_warnings():
    os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'


def init_realsense_pipeline():
    if rs is None:
        print("Ошибка: pyrealsense2 не найден.")
        return None

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
        print("RealSense D435I подключена успешно")
        return pipeline
    except Exception as e:
        print(f"Ошибка при запуске RealSense: {e}")
        return None


def create_dataset_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = f"dataset_{timestamp}"

    os.makedirs(os.path.join(dataset_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "depth"), exist_ok=True)

    print(f"Папка датасета создана: {dataset_path}")
    return dataset_path


def capture_realsense_frame(pipeline):
    try:
        # Ожидание кадров с таймаутом
        frames = pipeline.wait_for_frames(timeout_ms=5000)

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image, depth_image
    except Exception as e:
        print(f"Ошибка захвата кадра: {e}")
    return None, None


def save_frame(color_image, depth_image, dataset_path, frame_id):
    color_path = os.path.join(dataset_path, "color", f"{frame_id:06d}.jpg")
    depth_path = os.path.join(dataset_path, "depth", f"{frame_id:06d}.png")
    cv2.imwrite(color_path, color_image)
    cv2.imwrite(depth_path, depth_image)


def collect_dataset(pipeline, dataset_path, capture_interval=0.5):
    print("\n=== СБОР ДАТАСЕТА ===")
    print(f"Автосохранение каждые {capture_interval} секунд")
    print("Нажмите 'Q' для выхода\n")

    frame_id = 0
    last_capture_time = time.time()

    try:
        while True:
            color_image, depth_image = capture_realsense_frame(pipeline)

            if color_image is None:
                continue

            current_time = time.time()

            if current_time - last_capture_time >= capture_interval:
                save_frame(color_image, depth_image, dataset_path, frame_id)
                print(f"Сохранён кадр {frame_id:06d}")
                frame_id += 1
                last_capture_time = current_time

            display_image = color_image.copy()
            cv2.putText(display_image, f"Кадров: {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            time_until_next = max(0, capture_interval - (current_time - last_capture_time))
            cv2.putText(display_image, f"След кадр через: {time_until_next:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Датасет RealSense D435I', display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nСбор завершен. Всего сохранено: {frame_id} кадров")
        print(f"Путь: {dataset_path}")


def collect_dataset_with_yolo(pipeline, dataset_path, model, capture_interval=0.5):
    print("\n=== СБОР ДАТАСЕТА С YOLO ===")
    print(f"Автосохранение каждые {capture_interval} секунд")
    print("Нажмите 'Q' для выхода\n")

    frame_id = 0
    last_capture_time = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    try:
        while True:
            color_image, depth_image = capture_realsense_frame(pipeline)

            if color_image is None:
                continue

            current_time = time.time()

            if current_time - last_capture_time >= capture_interval:
                save_frame(color_image, depth_image, dataset_path, frame_id)
                print(f"Сохранён кадр {frame_id:06d}")
                frame_id += 1
                last_capture_time = current_time

            # Предикт модели
            results = model(color_image, verbose=False, device=device)
            annotated_image = results[0].plot()

            cv2.putText(annotated_image, f"Кадров: {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            time_until_next = max(0, capture_interval - (current_time - last_capture_time))
            cv2.putText(annotated_image, f"След кадр через: {time_until_next:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('YOLO + Датасет', annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nСбор завершен. Всего сохранено: {frame_id} кадров")


def main():
    suppress_warnings()

    pipeline = init_realsense_pipeline()
    if pipeline is None:
        return

    dataset_path = create_dataset_folder()

    print("\nВыберите режим сбора датасета:")
    print("1 - Только сохранение (без визуализации YOLO)")
    print("2 - С детекцией YOLO в реальном времени")

    choice = input("Ваш выбор: ").strip()

    capture_interval = 0.5

    if choice == "1":
        collect_dataset(pipeline, dataset_path, capture_interval)
    elif choice == "2":
        model_name = "yolov8n.pt"
        model = YOLO(model_name)
        if torch.cuda.is_available():
            model.to('cuda:0')
        collect_dataset_with_yolo(pipeline, dataset_path, model, capture_interval)
    else:
        print("Неверный выбор. Завершение работы.")
        pipeline.stop()


if __name__ == '__main__':
    main()
