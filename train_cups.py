from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='annotated_cups_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        patience=10,
        save=True,
        plots=True
    )
