import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy import interpolate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')

    return parser.parse_args()


def draw_points(img, points, color=(0, 0, 250)):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 3, color, cv2.FILLED, cv2.LINE_AA)


def draw_line_segments(img, points):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (127, 0, 0),
        (0, 127, 0),
        (0, 0, 127),
        (127, 127, 0)
    ]
    points = points.astype(np.int32)
    chin = points[0:17]
    left_brow = points[17:22]
    right_brow = points[22:27]
    nose1 = points[27:31]
    nose1 = np.concatenate((nose1, [points[33]]))
    nose2 = points[31:36]
    left_eye = points[36:42]
    left_eye = np.concatenate((left_eye, [points[36]]))
    right_eye = points[42:48]
    right_eye = np.concatenate((right_eye, [points[42]]))
    mouth = points[48:60]
    mouth = np.concatenate((mouth, [points[48]]))
    mouth_internal = points[60:68]
    mouth_internal = np.concatenate((mouth_internal, [points[60]]))
    lines = np.array([
        chin, left_brow, right_brow,
        nose1, nose2, left_eye,
        right_eye, mouth, mouth_internal
    ])
    for i, line in enumerate(lines):
        # tck, u = interpolate.splprep(line.transpose(), u=None, s=0.0, per=0)
        # u_new = np.linspace(u.min(), u.max(), 100)
        # xs, ys = interpolate.splev(u_new, tck, der=0)
        # line = np.stack([xs, ys]).transpose()
        cv2.polylines(
            img,
            np.int32([line]), False,
            colors[i], thickness=2, lineType=cv2.LINE_AA
        )


def main():
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, '*/*.jpg')))
    landmark_path = None
    landmarks = None
    for image_path in image_paths:
        dirname = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        new_landmark_path = os.path.join(dirname, 'landmarks.json')
        if landmark_path != new_landmark_path:
            if landmark_path is not None:
                with open(landmark_path, 'w') as f:
                    json.dump(landmarks, f)
            landmark_path = new_landmark_path
            with open(landmark_path, 'r') as f:
                landmarks = json.load(f)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_landmarks = np.array(landmarks[basename])
        img_landmarks *= [img.shape[1], img.shape[0]]

        txt = "Press 'Space' or 'N'"
        (x_size, y_size), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
        cv2.putText(
            img,
            txt,
            (img.shape[1] // 2 - x_size // 2, int(img.shape[0] * 0.9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA
        )
        cv2.putText(
            img,
            txt,
            (img.shape[1] // 2 - x_size // 2, int(img.shape[0] * 0.9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (250, 250, 250), thickness=1, lineType=cv2.LINE_AA
        )
        # draw_points(img, img_landmarks)
        draw_line_segments(img, img_landmarks)

        cv2.imshow('Validate', img)

        end = False
        while True:
            key = cv2.waitKey(0)
            if key in {ord('n'), ord('N')}:
                del landmarks[basename]
                os.remove(image_path)
                break
            elif key == 32:
                break
            elif key == 27:
                with open(landmark_path, 'w') as f:
                    json.dump(landmarks, f)
                end = True
                break

        if end:
            break


if __name__ == '__main__':
    main()
