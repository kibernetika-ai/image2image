import argparse
import glob
import json
import os

import cv2
import numpy as np

import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')

    return parser.parse_args()


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
        # common.draw_points(img, img_landmarks)
        common.draw_line_segments(img, img_landmarks, interpolation=False)

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

    if landmark_path is not None:
        with open(landmark_path, 'w') as f:
            json.dump(landmarks, f)


if __name__ == '__main__':
    main()
