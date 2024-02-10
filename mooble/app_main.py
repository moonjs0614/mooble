from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)

# 웹캠 활성화 함수
def generate_frames():
    camera = cv2.VideoCapture(0)  # 0번 카메라
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # 웹페이지 렌더링

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)