from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='conf/dataset.yaml',
   imgsz=640,
   epochs=8,
   name='finetuned'
)
results = model.val()