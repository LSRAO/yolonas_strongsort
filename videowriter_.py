import cv2 as cv

path = './clip_13.mp4'
output_path = './output/try.mp4'
cap = cv.VideoCapture(path)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

forcc = cv.VideoWriter_fourcc(*"mp4v")
video = cv.VideoWriter(output_path,
                       forcc,
                       fps,
                       (frame_width,frame_height))

while True:
    ret, frame = cap.read()

    if ret:
        video.write(frame)
    else:
        break

cap.release()
video.release()
cv.destroyAllWindows()