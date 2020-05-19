import argparse
import json
import logging
import os

import cv2
import dlib
import numpy as np
from ml_serving.drivers import driver


LOG = logging.getLogger(__name__)
threshold = 0.9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-model')
    parser.add_argument('--landmarks-model')
    parser.add_argument('--input')
    parser.add_argument('--output', default='output')

    return parser.parse_args()


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    ymin = int(max([box[1] - h * margin, 0]))
    ymax = int(min([box[3] + h * margin, img.shape[0]]))
    xmin = int(max([box[0] - w * margin, 0]))
    xmax = int(min([box[2] + w * margin, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_landmarks(model, frame, box):
    box = box.astype(np.int)
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    shape = model(frame, rect)
    shape = shape_to_np(shape)

    return shape


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
    cropped_ratio = 0.2
    boxes = {}
    all_landmarks = {}
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame_num += 1

        face_boxes = get_boxes(face_driver, frame)

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
        landmarks = get_landmarks(landmarks_driver, cropped_frame, np.array(new_box)).astype(float)
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
