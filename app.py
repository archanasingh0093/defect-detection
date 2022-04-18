from flask import Flask, request, render_template, Response
import numpy as np
import cv2
import json
import time
from utilities import load_model, load_label_map, show_inference, parse_output_dict
from custom_np_encoder import NumpyArrayEncoder


model_path = "models/export/saved_model"
labels_path = "data/label_map.pbtxt"

vis_threshold = 0.6
max_boxes = 20

detection_model = load_model(model_path)
category_index = load_label_map(labels_path)

app = Flask(__name__)

video = cv2.VideoCapture(0)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frames():
    """Video streaming generator function."""
    while True:
        stime = time.time()
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), None, .5, .5)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            nparr = np.frombuffer(frame, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # do inderence on the image and get detections and image data with overlay
            output_dict, image_np = show_inference(detection_model, img, category_index, vis_threshold, max_boxes)
            parsed_output_dict = parse_output_dict(output_dict, category_index)

            parsed_output_dict.update({"image data": image_np})

            # add the size of the image in the response
            parsed_output_dict.update({"image size": "size={}x{}".format(img.shape[1], img.shape[0])})

            # build a response dict to send back to client
            response = parsed_output_dict
            img_array = np.asarray(response["image data"])
            cv2.imwrite("filename.png", img_array)
            img = cv2.imread("filename.png")
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            
            # encode response
            # response_encoded = json.dumps(response, cls=NumpyArrayEncoder)
            print('FPS {:.1f}'.format(1/ (time.time() - stime)))
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
