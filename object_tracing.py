import os
import torch
import time
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import colorsys
import json

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.pipelines.pipelines import DetectionPipeline

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from strong_sort.sort.tracker import Tracker

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def detection_in_img(img_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_nas = models.get("yolo_nas_s", pretrained_weights="coco")
    model_predictions  = list(yolo_nas.predict(img_path,conf = 0.7)._images_prediction_lst)

    bboxes_xyxy = model_predictions[0].prediction.bboxes_xyxy.tolist()
    confidence = model_predictions[0].prediction.confidence.tolist()
    labels = model_predictions[0].prediction.labels.tolist()
    lengths = [len(labels),len(confidence),len(bboxes_xyxy)] 
    labels = [int(label) for label in labels]

    person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
    person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
    person_labels = [label for label in labels if label == 0]

    model_predictions_details = {}
    model_predictions_details['bboxes_xyxy'] = bboxes_xyxy[0]
    model_predictions_details['confidence'] = confidence[0]
    model_predictions_details['label'] = labels[0]
    model_predictions_details['lengths'] = lengths

    with open('model_predictions_details.json', 'w') as json_file:
        json.dump(model_predictions_details, json_file)
    
    model_predictions_details_for_person = {}
    model_predictions_details_for_person['bboxes_xyxy'] = person_bboxes_xyxy
    model_predictions_details_for_person['confidence'] = person_confidence
    model_predictions_details_for_person['label'] = person_labels
    model_predictions_details_for_person['lengths'] = [len(person_labels),len(person_confidence),len(person_bboxes_xyxy)]
    
    with open('model_predictions_details_for_person.json', 'w') as json_file:
        json.dump(model_predictions_details_for_person, json_file)
    
    # plot the filtered person predictions, confidence , label on the image
    img = cv2.imread(img_path)
    for bbox in person_bboxes_xyxy:
        x1,y1,x2,y2 = bbox
        color = (0,255,0)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        text_color = (0,0,0)
        cv2.putText(img,f'{str(person_labels[0])}',(int(x1)+10,int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5, text_color,2)
        
    # save the image
    cv2.imwrite('output/out.jpg',img)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    









    # model_predictions.save(f'D:\Ksquarez\Yolonas and Deep Sort\output\out.jpg')

def get_prediction(image_in,pipeline,model):
    preprocessed_image, processing_metadata = pipeline.image_processor.preprocess_image(image = image_in.copy())
    
    with torch.no_grad():
        torch_input = torch.Tensor(preprocessed_image).unsqueeze(0).to(device)
        model_output = model(torch_input)
        prediction = pipeline._decode_model_output(model_output=model_output,model_input = torch_input)
        print(prediction)
        
    return pipeline.image_processor.postprocess_predictions(predictions = prediction[0], metadata = processing_metadata)
        
    
def get_color(number):
    hue = number * 30%180
    saturation = number * 103 % 256
    value = number * 50 % 256
    
    color = colorsys.hsv_to_rgb(hue / 179, saturation / 255, value / 255)
    
    return [int(c*255) for c in color]


def loading_models():
    
    strong_sort_weights = "/home/ksuser/Documents/strongsort/yolonas_deepsort/osnet_x0_25_msmt17.pt"
    tracker = StrongSORT(model_weights=strong_sort_weights,max_age=70,device=device)    
    yolo_nas = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").cuda()
    return tracker,yolo_nas

def tracking(video_path,):
    tracker,yolo_nas = loading_models()
    
    pipeline = DetectionPipeline(
        model = yolo_nas,
        image_processor = yolo_nas._image_processor,
        post_prediction_callback = yolo_nas.get_post_prediction_callback(iou = 0.55, conf = 0.75),
        class_names = yolo_nas._class_names[0],
    )
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    isfile = True
    while isfile:
        num = 0
        output_list = os.listdir('output/')
        output_path = f'output/output_{num}.mp4'
        if output_path in output_list:
            num+=1
        else:
            isfile = False
            
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    i = 0
    counter, fps, elasped = 0,0,0
    start_time = time.process_time()

    while True:
        h,w = 0,0
        ret, frame = cap.read()

        if ret:
            # frame = cv2.resize(frame)

            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = og_frame.copy()

            with torch.no_grad():
                conf_threshold = 0.7
                device = 0

                pred = get_prediction(frame,pipeline,model=yolo_nas)
                labels = pred.labels
                
                class_id_mapping = {class_label: class_id for class_id , class_label in enumerate(pipeline.class_names)}
                default_class_id = 0
                classes = np.array([class_id_mapping.get(class_label,default_class_id) for class_label in labels])
                

                bboxes_xyxy = detection_pred[0].prediction.bboxes_xyxy.tolist()
                confidence = detection_pred[0].prediction.confidence.tolist()

                class_name = ['person']

                # person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
                # person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
                # person_labels = [label for label in labels if label == 0]

                bboxes_xywh = []
                for bbox in pred.bboxes_xyxy:
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    bboxes_xywh.append(bbox_xywh)
                
                bboxes_xywh = np.array(bboxes_xywh)

                tracks = tracker.update(bboxes_xywh,pred.confidence,classes,og_frame)

                for track in tracker.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    track_id = track.track_id
                    hits = track.hits
                    bbox = track.to_tlbr()
                    # class_name = track.get_class()
                    x1,y1,x2,y2 =bbox
                    w = x2 -x1
                    h = y2 -y1

                    bbox_xywh = (x1,y1,w,h)
                    color = get_color(track_id * 15)

                    cv2.rectangle(og_frame,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)

                    text_color = (255,0,0)
                    cv2.putText(og_frame,f'{class_name[0]} - {str(track_id)}',(int(x1)+10,int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5, text_color,2)
                
                current_time = time.process_time()
                elasped = (current_time - start_time)
                counter +=1
                if elasped > 1:
                    fps = counter/elasped
                    counter = 0
                    start_time = current_time
                
                cv2.putText(og_frame,
                            f'FPS: {str(round(fps,2))}',
                            (10,int(h)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255),
                            2,
                            cv2.LINE_AA)
                
                frames.append(og_frame)

                
                out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

                # cv2.imshow('Video', og_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        else:
            break        
    
    cap.release()
    out.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':

    # strong_sort_weights = r'strong_sort/deep/checkpoint/ckpt.t7'
    #change this
    video_path = r'/home/ksuser/LS/clips/clip_13.mp4'
    
    tracking(video_path)
    
    # for Image
    # img_path = r"images/office.PNG"
    # detection_in_img(img_path)
