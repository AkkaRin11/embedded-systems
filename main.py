import os
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def suppress_warnings():
    os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'


def check_gpu_usage():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    print(f"GPU устройства: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Текущий GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA память: {torch.cuda.memory_allocated() / 1e9:.1f}GB / "
              f"{torch.cuda.memory_reserved() / 1e9:.1f}GB")


def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("Нет подключенных RealSense камер")
        return []
    for i, dev in enumerate(devices):
        print(f"{i}: {dev.get_info(rs.camera_info.name)}")
    return devices


def start_realsense_pipeline(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def get_frame_from_pipeline(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    # Convert to numpy array for OpenCV
    frame = np.asanyarray(color_frame.get_data())
    return frame


def load_yolo_model(model_path="best.pt"):
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda:0')
        print(f"YOLO устройство: {next(model.model.parameters()).device}")
        print(f"Модель загружена: {model_path}")
    else:
        print("YOLO на CPU")
    return model


def detect_cups_only(model, frame, conf_threshold=0.5):
    results = model(frame,
                    conf=conf_threshold,
                    verbose=False,
                    device='cuda:0' if torch.cuda.is_available() else 'cpu')
    return results


def draw_yolo_detections(frame, results):
    annotated_frame = results[0].plot()
    return annotated_frame


def analyze_cup_position(frame, results):
    """
    Анализирует положение первой обнаруженной кружки и выводит направление поворота.
    """
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, _, x2, _ = box.xyxy[0]
        cup_center_x = (x1 + x2) / 2
        screen_center_x = frame.shape[1] / 2
        tolerance = 30
        if cup_center_x < screen_center_x - tolerance:
            print("Нужно повернуться влево")
        elif cup_center_x > screen_center_x + tolerance:
            print("Нужно повернуться вправо")
        else:
            print("Кружка находится по центру")


def display_realsense_yolo(pipeline, model):
    check_gpu_usage()
    try:
        while True:
            frame = get_frame_from_pipeline(pipeline)
            if frame is None:
                print("Не удалось захватить кадр")
                continue

            results = detect_cups_only(model, frame, conf_threshold=0.5)
            analyze_cup_position(frame, results)
            annotated_frame = draw_yolo_detections(frame, results)

            cv2.imshow('Детекция кружек (RealSense)', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        check_gpu_usage()
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    suppress_warnings()
    list_realsense_devices()

    pipeline = start_realsense_pipeline()
    model = load_yolo_model()

    display_realsense_yolo(pipeline, model)


if __name__ == '__main__':
    main()
