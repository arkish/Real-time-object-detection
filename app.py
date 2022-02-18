from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for,  Response
from werkzeug.utils import secure_filename
# , send_from_directory
import os
import subprocess
import cv2



import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from detect import run

app = Flask(__name__)                 

uploads_dir = os.path.join(app.instance_path, 'uploads')

# camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)

os.makedirs(uploads_dir, exist_ok=True)

# Initialize




@app.route("/")
def hello_world():
    return render_template('index.html')

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # print(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(subprocess.run(['python3.7', 'detect.py', '--source', '0']), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(run(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detectObject", methods=['POST'])
def detect():
    if not request.method == "POST":
        return
    video = request.files['video']
    video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    print(video)
    subprocess.run("ls")
    # subprocess.run(['python3.7', 'detect.py', '--source', os.path.join(uploads_dir, secure_filename(video.filename))])
    subprocess.run(['python3.7', 'detect.py', '--source', '0'])

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    obj = secure_filename(video.filename)
    return obj

@app.route('/return-files', methods=['GET'])
def return_file():
    obj = request.args.get('obj')
    loc = os.path.join("runs/detect", obj)
    print(loc)
    try:
        return send_file(os.path.join("runs/detect", obj), attachment_filename=obj)
        # return send_from_directory(loc, obj)
    except Exception as e:
        return str(e)

# @app.route('/display/<filename>')
# def display_video(filename):
# 	#print('display_video filename: ' + filename)
# 	return redirect(url_for('static/video_1.mp4', code=200))


if __name__ == '__main__':
	app.run(debug = True)
    # cors = require('cors')  #use this
    # app.use(cors()) #and this