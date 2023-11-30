from ultralytics import YOLO
import cv2


# Load a model
model = YOLO('yolov8m-pose.pt')  # load an official model

# Predict with the model
results = model(r'D:\PycharmPojects\Pose_Estimation_to_Animation\bus.jpg')
#image = cv2.imread(r'D:\PycharmPojects\Pose_Estimation_to_Animation\bus.jpg')

image = results[0].plot()
print(results[0])

cv2.imshow("", image)
cv2.waitKey(0)