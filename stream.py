#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# TODO: Add option to start streaming server

# TODO: Create ZMQ PAIR server as type of renderer

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_streaming_server():
    app.run(host='0.0.0.0', debug=True)

if __name__ == '__main__':
    run_streaming_server()

