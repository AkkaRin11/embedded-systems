import cv2
import os
import torch
from ultralytics import YOLO


def suppress_warnings():
    os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'


def check_gpu_usage():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    print(f"GPU устройства: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Текущий GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA память: {torch.cuda.memory_allocated() / 1e9:.1f}GB / {torch.cuda.memory_reserved() / 1e9:.1f}GB")


def list_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ret, _ = cap.read()
        if ret:
            available.append(i)
            print(f"Камера {i} работает")
        cap.release()
    return available


def capture_camera_stream(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {camera_index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


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
    results = model(frame, conf=conf_threshold, verbose=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    return results


def draw_yolo_detections(frame, results):
    annotated_frame = results[0].plot()
    return annotated_frame
def analyze_cup_position(frame, results):
    """
    Анализирует положение первой обнаруженной кружки и выводит в консоль направление поворота.
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



def display_camera_stream_yolo(cap, model):
    check_gpu_usage()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр")
            break

        results = detect_cups_only(model, frame, conf_threshold=0.5)
        analyze_cup_position(frame, results)
        annotated_frame = draw_yolo_detections(frame, results)

        cv2.imshow('Детекция стаканов (Своя модель)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    check_gpu_usage()
    cap.release()
    cv2.destroyAllWindows()


def main():
    suppress_warnings()
    print(list_cameras())

    camera_index = 2
    cap = capture_camera_stream(camera_index)
    if cap is None:
        return

    model = load_yolo_model()
    display_camera_stream_yolo(cap, model)


if __name__ == '__main__':
    main()
