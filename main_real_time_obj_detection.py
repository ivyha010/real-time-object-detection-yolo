from ultralytics import YOLO
import cv2
import keyboard

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolo11n.pt") # use the pretrained yolo model
#model = YOLO("./runs/detect/train/weights/best.pt")  # load your custom model, which was trained on your custom dataset(
#use COCO8 dataset as an example for training a custom model)

while True:
    success, img = cap.read()
    results = model.predict(source=img, show=True)

    if keyboard.is_pressed('q'): # stop when the 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()