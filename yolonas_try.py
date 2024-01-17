import super_gradients
import cv2
import numpy as np
from PIL import Image
import torch 
import torchvision 
from torchvision.io import read_image 
import torchvision.transforms as T 

yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco")
# print(dir(yolo_nas))

# print(cv2.dnn.readNet('yolo_nas_s.pt'))
# image = cv2.imread('./beatles.jpg')

image = Image.open('./beatles.jpg',)
image = np.asarray(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = read_image('./beatles.jpg')

model_predictions  = list(yolo_nas.predict(image)._images_prediction_lst)

prediction = model_predictions[0].prediction        # One prediction per image - Here we work with 1 image so we get the first.

# bboxes = prediction.bboxes_xyxy                     # [[Xmin,Ymin,Xmax,Ymax],..] list of all annotation(s) for detected object(s) 
# class_names = prediction.class_names                # ['Class1', 'Class2', ...] List of the class names
# class_name_indexes = prediction.labels.astype(int)  # [2, 3, 1, 1, 2, ....] Index of each detected object in class_names(corresponding to each bounding box)
# confidences =  prediction.confidence.astype(float)  # [0.3, 0.1, 0.9, ...] Confidence value(s) in float for each bounding boxes

bboxes_xyxy = prediction.bboxes_xyxy.tolist()
confidence = prediction.confidence.tolist()
labels = prediction.labels.tolist()
lengths = [len(labels),len(confidence),len(bboxes_xyxy)] 
labels = [int(label) for label in labels]

person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
person_labels = [label for label in labels if label == 0]
person_labels_names = ['person' for x in labels if x==0]

def draw_bbox_with_info(image,person_bboxes_xyxy,person_confidence,class_names):
    # Draw bounding box
    
    result = image.copy()
    for bbox,class_name,conf in zip(person_bboxes_xyxy,class_names,person_confidence):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(result, (x, y), (w, h), (0, 255, 0), 2)

        # Display class and confidence on top of the bounding box
        text = f'{class_name}: {conf:.2f}'
        print(text)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x
        text_y = y - 5 if y - 5 > 0 else y + 20

        # Create a black background for the text
        overlay = np.zeros_like(image)
        cv2.rectangle(overlay, (text_x - 2, text_y - text_size[1] - 2), (text_x + text_size[0] + 2, text_y + 2), (255, 0, 0), -1)
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        # Combine the original image and the overlay
        result = cv2.addWeighted(result, 1, overlay, 0.7, 0)

    return result
# print(person_labels_names)
img = draw_bbox_with_info(image,person_bboxes_xyxy,person_confidence,person_labels_names)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('output/out.jpg',img)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
