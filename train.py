from  ultralytics import YOLO

model = YOLO('runs/detect/train28/weights/last.pt')

model.train(data = 'datasets/Real-Time-Accident-Detection-1/data.yaml',
          epochs=20, imgsz=640, batch=64, workers= 1, device = 0, save_period= 2,
          cls= 15.5, box=0.4 ,rect=True, cos_lr=True, close_mosaic=20
          )