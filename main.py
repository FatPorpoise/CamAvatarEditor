from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

video = cv2.VideoCapture('C:/Users/FatPorpoise/Downloads/ja.mp4')
frames = []
while True:
    read, frame = video.read()
    if not read:
        break
    frames.append(frame)
    # Display the frame
    cv2.imshow('Video', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
frames = np.array(frames)
video.release()

cv2.destroyAllWindows()

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(0, 0, 255), width=6)

    # Add to frame list
    frames_tracked.append(frame_draw)
    cv2.imshow('Video', np.asarray(frame_draw))

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('\nDone')


dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 30.0, dim)
for frame in frames_tracked:
    video_tracked.write(np.array(frame))

video_tracked.release()

cv2.destroyAllWindows()
