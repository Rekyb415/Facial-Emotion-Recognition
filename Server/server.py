from flask import Flask, Response
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')
model = tf.keras.models.load_model('model.h5')

labels = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contemp"]


def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    settings = {
        'scaleFactor': 1.3,
        'minNeighbors': 5,
        'minSize': (50, 50)
    }

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_detection.detectMultiScale(gray, **settings)

            for x, y, w, h in detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 135, 66), 2)
                cv2.rectangle(frame, (x, y), (x + w // 3, y + 20), (245, 135, 66), -1)
                face = gray[y + 5:y + h - 5, x + 20:x + w - 20]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0

                predictions = model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
                state = labels[predictions]
                cv2.putText(frame, state, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
