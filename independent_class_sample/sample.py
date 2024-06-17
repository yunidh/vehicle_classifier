from vehicle_classifier_class import VehicleClassifier
import cv2

model_path = "modelv3_3OP.pth"
classifier = VehicleClassifier(model_path)
cv_image = cv2.imread("test_images/1.jpg")
pred = classifier.predict(cv_image)
pred  # -1:no wheeler, 0: two wheeler, 1: four wheeler
print(f"Prediction: {pred}")

class_labels = ["no wheeler", "two wheeler", "four wheeler"]
print(f"pred label: {class_labels[pred]}")
