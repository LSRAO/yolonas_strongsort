import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "clip_13.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        
        height, width = frame.shape[:2]
        max_height = 800
        max_width = 1200
        scale = min(max_height/height, max_width/width)

        resized_image = cv2.resize(frame, None, fx=scale, fy=scale)
        # Run YOLOv8 inference on the frame
        results = model(resized_image, classes=0,iou=0.9,half=True)

        # # Visualize the results on the frame
        # person_bbox = []
        # person_conf = []
    
        # for cls,bbox,conf in zip(results[0].boxes.cls,results[0].boxes.data,results[0].boxes.conf):
        #     if cls == 0:
        #         person_bbox.append(bbox)
        #         person_conf.append(conf)

        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference",annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()