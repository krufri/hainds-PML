import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from math import sqrt
import time
from typing import List
import datetime

import cv2 as cv
import numpy as np
import pandas as pd
import mediapipe as mp
from cmath import e

from model import HandGestureClassifier
from model import PointHistoryClassifier

import tensorflow as tf
from tensorflow import keras
from face_toolbox_keras.models.parser import face_parser

import threading
import os

from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=500)
    parser.add_argument("--height", help='cap height', type=int, default=500)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

# Global placeholder
emoji_detec="default"
color_detec="default"
moving_detec="default"


def gen_frames():
    # Argument parsing
    args = get_args()

    cv.ocl.setUseOpenCL(False)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Face detection and skin extraction preprocessing initialization
    global model_initialized
    model_initialized = False
    global execute_skin_color_detection
    execute_skin_color_detection = True
    start_time = datetime.datetime.now()
    faceCascade = cv.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    model = initialize_skin_detection_model()
    prs = face_parser.FaceParser()

    # Model load
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    handgesture_classifier = HandGestureClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/handgesture_classifier/emoji_classifier_labels.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    while True:
        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Emoji var
        emoji_text="default"

        # Face detection implementation
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for x,y,w,h in faces:
            cv.rectangle(debug_image, (x, y), (x + w, y + h),(255,0,0),2)

        if(len(faces) > 0):
            current_time = datetime.datetime.now()
            timedelta = current_time - start_time
        else:
            timedelta = start_time - start_time

        if len(faces) > 0 and execute_skin_color_detection == True and model_initialized == True and timedelta.seconds > 2:
            execute_skin_color_detection = False
            face_points = faces[0]
            x,y,w,h = face_points[0], face_points[1], face_points[2], face_points[3],
            face_cropped = debug_image[y:y+h, x:x+w]
            thread = threading.Thread(target=classify_skin_color, args=[model, prs, face_cropped])
            thread.start()
        elif len(faces) == 0 and execute_skin_color_detection == True and model_initialized == True and timedelta.seconds > 2:
            execute_skin_color_detection = False


        #  Check if hands are shown
        if results.multi_hand_landmarks is not None:

            # Check if two hands are shown
            if len(results.multi_handedness) == 2:
                two_hands_landmark_list = list()

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    temp_landmark_list = copy.deepcopy(landmark_list)

                    # Convert to relative coordinates
                    base_x, base_y = 0, 0
                    for index, landmark_point in enumerate(temp_landmark_list):
                        if index == 0:
                            base_x, base_y = landmark_point[0], landmark_point[1]

                        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    two_hands_landmark_list.extend(pre_processed_landmark_list)

                 # Hand sign classification
                hand_sign_id = handgesture_classifier(two_hands_landmark_list)
                point_history.append([0, 0])

                emoji_text = "Both:" + keypoint_classifier_labels[hand_sign_id]


            else:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    temp_landmark_list = copy.deepcopy(landmark_list)

                    # Convert to relative coordinates
                    base_x, base_y = 0, 0
                    for index, landmark_point in enumerate(temp_landmark_list):
                        if index == 0:
                            base_x, base_y = landmark_point[0], landmark_point[1]

                        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)

                    one_hand_landmark_list = [0] * 42
                    one_hand_landmark_list.extend(pre_processed_landmark_list)

                    # Hand sign classification
                    hand_sign_id = handgesture_classifier(one_hand_landmark_list)

                    if hand_sign_id == 3:  # Wave & Raising
                        point_history.append(landmark_list[12])
                    elif hand_sign_id == 19:  # Clapp
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Moving hand classification
                    finger_gesture_id = 1
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()

                    moving_gesture=point_history_classifier_labels[most_common_fg_id[0][0]]
                    #print("Moving gesture: " + moving_gesture)
                    global moving_detec
                    moving_detec=moving_gesture

                    emoji_text=handedness.classification[0].label[0:]+':'+keypoint_classifier_labels[hand_sign_id]


        else:
            point_history.append([0, 0])

        global emoji_detec
        emoji_detec=emoji_text

        success, buffer = cv.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--image\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoints
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def initialize_skin_detection_model():
    model = keras.models.load_model("skin-color-detection/model_training/faces_model_improved.h5")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    global model_initialized
    model_initialized = True
    return model

def classify_skin_color(model, face_parser, image):
    global model_initialized
    model_initialized = False
    class_names = ['dark', 'light', 'medium', 'veryDark', 'veryLight']
    resizedImage = resize_image(image)
    mask = face_parser.parse_face(resizedImage)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            for k in range(len(mask[0][0])):
                if mask[i][j][k] != 1:
                    mask[i][j][k] = 0
    newmask = mask[0]
    newImg = cv.bitwise_and(resizedImage,resizedImage,mask=newmask)
    resizedFace = cv.resize(newImg, (250, 187), interpolation= cv.INTER_LINEAR)
    img_array = keras.preprocessing.image.img_to_array(resizedFace)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    # Parse to global
    global color_detec
    color_detec=class_names[np.argmax(score)]
    model_initialized = True

def resize_image(im, max_size=450):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        return cv.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

def gen_emoji():
    # Read emoji_index

    # Parse
    global emoji_detec
    emoji=emoji_detec

    global color_detec
    color=color_detec

    # Text to icon
    # Remove pretext
    if emoji[0]=='L':
        emoji=emoji[5:]
    elif emoji[0]=='R':
        emoji=emoji[6:]
    elif emoji[0]=='B':
        emoji=emoji[5:]
    else:
        emoji=""

    global moving_detec

    df = pd.read_csv('emoji_index.csv')
    #if Movment was detect use specific emoji
    if moving_detec=="Wave":
        emoji_df=df.loc[(df['gesture'] == "wave") & (df['skin_color'] == color)]
        emoji=str(emoji_df['icon'].values)
    elif moving_detec=="Raising":
        emoji_df=df.loc[(df['gesture'] == "raising") & (df['skin_color'] == color)]
        emoji=str(emoji_df['icon'].values)
    elif moving_detec=="Clapping":
        emoji_df=df.loc[(df['gesture'] == "clapping") & (df['skin_color'] == color)]
        emoji=str(emoji_df['icon'].values)
    else:
        emoji_df=df.loc[(df['gesture'] == emoji) & (df['skin_color'] == color)]
        if len(emoji_df) == 0:
            emoji=""
        else:
            if(emoji_df['gesture'].values == "heart" or emoji_df['gesture'].values == "index"):
                emoji= str(emoji_df['icon'])
                emoji=emoji[7:-26]

                return emoji
            if(emoji_df['gesture'].values == "finger-heart"):
                emoji= str(emoji_df['icon'])
                emoji=emoji[6:-25]
                return emoji

            emoji= str(emoji_df['icon'].values)

    return emoji[2:-2]

# flask
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html", emoji = " ")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=image')

@app.route("/emoji_feed", methods=['GET'])
def emoji_feed():
    emoji = gen_emoji()
    return jsonify({ 'emoji': emoji })

@app.route("/detect_skin_color")
def detect_skin_color():
    global execute_skin_color_detection
    execute_skin_color_detection = True
    return ("nothing")

@app.route("/manual_color_selection", methods=['POST'])
def manual_color_selection():
    global color_detec
    color_detec = request.form['skinColorSelection']
    emoji = gen_emoji()
    return jsonify({ 'emoji': emoji })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)