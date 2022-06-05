from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import tensorflow as  tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

emotion_dict = {0:"Angry", 1: "Disgusted", 2:"Fearfull", 3:"Happy", 4:"Nutral", 5:"Sad", 6:"Surprised"}
emotion_model = model_from_json(open("model/emotion_model.json", "r").read())  #load model

emotion_model.load_weights('model/emotion_model.h5')                            #load weights

cap = cv2.VideoCapture(0)

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = cap.read()  # read the camera frame

        if not success:
            break
        else:
            face_detector = cv2.CascadeClassifier(
                'C:\\Users\\POWER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect facesavailable on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

            resized_img = cv2.resize(frame, (1000, 700))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)