import cv2
import requests
import numpy as np

url = 'http://localhost:5000/video_feed'

camera = cv2.VideoCapture(url)

while True:
    ret, frame = camera.read()
    cv2.imshow('Facial Expression', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
