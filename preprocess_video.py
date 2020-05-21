import argparse
import json
import logging
import os

import cv2
import dlib
import numpy as np
from ml_serving.drivers import driver

import common


LOG = logging.getLogger(__name__)
threshold = 0.9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-model')
    parser.add_argument('--landmarks-model')
    parser.add_argument('--input')
    parser.add_argument('--output', default='output')

    return parser.parse_args()


def draw_points(img, points, color=(0, 0, 250)):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 3, color, cv2.FILLED, cv2.LINE_AA)


def load_models(face_path, shape_path):
    drv = driver.load_driver('openvino')
    face_driver = drv()
    face_driver.load_model(face_path)

    landmarks_driver = load_shape_model(shape_path)

    return face_driver, landmarks_driver


def load_shape_model(shape_path):
    return dlib.shape_predictor(shape_path)


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)

    face_driver, landmarks_driver = load_models(args.face_model, args.landmarks_model)
    LOG.info('Models loaded.')

    basename = os.path.splitext(os.path.basename(args.input))[0]
    dirname = os.path.join(args.output, basename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    vc = cv2.VideoCapture(args.input)
    frame_num = -1
    cropped_ratio = 0.05
    boxes = {}
    all_landmarks = {}
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame_num += 1

        face_boxes = common.get_boxes(face_driver, frame)

        if len(face_boxes) != 1:
            continue

        box = face_boxes[0]

        w = box[2] - box[0]
        h = box[3] - box[1]
        new_box = [
            int(max(round(box[0] - cropped_ratio * w), 0)),
            int(max(round(box[1] - cropped_ratio * h), 0)),
            int(min(round(box[0] + w * (1.0 + cropped_ratio)), frame.shape[1])),
            int(min(round(box[1] + h * (1.0 + cropped_ratio)), frame.shape[0])),
        ]
        file_name = os.path.join(dirname, f'{frame_num:05d}.jpg')
        cropped_frame = frame[new_box[1]:new_box[3], new_box[0]:new_box[2]]
        cv2.imwrite(file_name, cropped_frame)
        new_box = [
            max(box[0] - new_box[0], 0),
            max(box[1] - new_box[1], 0),
            min(new_box[2] - box[0], cropped_frame.shape[1]),
            min(new_box[3] - box[1], cropped_frame.shape[0]),
        ]
        landmarks = common.get_landmarks(landmarks_driver, cropped_frame, np.array(new_box)).astype(float)
        # draw_points(cropped_frame, landmarks)
        # cv2.imshow('img', cropped_frame)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     break

        # Get relative coords
        landmarks[:, 0] = landmarks[:, 0] / cropped_frame.shape[1]
        landmarks[:, 1] = landmarks[:, 1] / cropped_frame.shape[0]

        boxes[f'{frame_num:05d}.jpg'] = new_box
        all_landmarks[f'{frame_num:05d}.jpg'] = landmarks.tolist()

        if (frame_num + 1) % 100 == 0:
            LOG.info(f'Processed {frame_num+1} frames.')

    with open(os.path.join(dirname, 'boxes.json'), 'w') as f:
        f.write(json.dumps(boxes, indent=2))
    with open(os.path.join(dirname, 'landmarks.json'), 'w') as f:
        f.write(json.dumps(all_landmarks, indent=2))


if __name__ == '__main__':
    main()
