import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # В версии 8.2+ поддерживается такое же API обучения.
    # Для Python 3.8 важно убедиться, что torch совместим с CUDA, если она используется.
    model = YOLO('yolov8n.pt')

    # Определение устройства
    device = 0 if torch.cuda.is_available() else 'cpu'

    results = model.train(
        data='annotated_cups_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,  # Рекомендуется 0 для Windows во избежание проблем с мультипроцессорностью
        device=device,
        patience=10,
        save=True,
        plots=True
    )

    print("Обучение завершено.")
