import cv2
import requests
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

url = 'http://localhost:5000/video_feed'

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(url)
        self.image = Image()

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Cập nhật khung hình 30 lần mỗi giây

        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)  # Lật ảnh theo trục y để hiển thị đúng trên Kivy
            buf = frame.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()
