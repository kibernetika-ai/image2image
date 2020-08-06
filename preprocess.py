import argparse
import glob
import logging
import os
import random
import shutil
import tempfile
import threading

from concurrent import futures
import cv2
import face_alignment
from ml_serving.drivers import driver
import numpy as np
import torch
import youtube_dl

import common


face_model_path = (
    '/opt/intel/openvino/deployment_tools/intel_models'
    '/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--output', default='output')
    parser.add_argument('--cookie')
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


def intersect_area(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    x2 = min(rect1[2], rect2[2])
    y1 = max(rect1[1], rect2[1])
    y2 = min(rect1[3], rect2[3])
    if x1 >= x2:
        return 0
    if y1 >= y2:
        return 0

    inter_area = (x2 - x1) * (y2 - y1)
    box_a_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    box_b_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    return inter_area / float(box_a_area + box_b_area - inter_area)


def draw_points(img, points, color=(0, 0, 250)):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 3, color, cv2.FILLED, cv2.LINE_AA)


class VOXCeleb(object):
    def __init__(self, data_dir, face_driver, cookiefile=None, workers=4):
        self.data_dir = data_dir
        use_cuda = torch.cuda.is_available()
        use_device = 'cuda' if use_cuda else 'cpu'

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=use_device)

        self.face_driver = face_driver
        self.k = 45
        self.min_face_size = 80
        self.min_image_size = self.min_face_size * 1.8
        self.max_workers = workers
        if cookiefile and not os.path.exists(cookiefile):
            # Raise an error.
            os.stat(cookiefile)

        self.cookie = cookiefile
        self.sem = threading.Semaphore(value=self.max_workers)
        self.stopped = False
        # structure: {root}/txt/{idXXXX}/{video_id}/{XXXX}.txt

    def process_videos(self, output_dir):
        videos = glob.glob(os.path.join(self.data_dir, '*/*/*'))
        # random.shuffle(videos)
        pool = futures.ThreadPoolExecutor(max_workers=self.max_workers)
        for i, video_dir in enumerate(videos):
            video_id = video_dir.split('/')[-1]
            if video_dir.endswith('.mp4'):
                video_url = '/'.join(video_dir.split('/')[-3:-1:1])
            else:
                video_url = f'https://www.youtube.com/watch?v={video_id}'

            LOG.info(f'[{i}/{len(videos)}] Start processing video {video_url}...')
            self.sem.acquire()
            if self.stopped:
                return

            if video_dir.endswith('.mp4'):
                pool.submit(self.process_video, video_dir, os.path.join(output_dir, video_id), self.fa)
            else:
                pool.submit(self.process_text_dir, video_dir, video_url, os.path.join(output_dir, video_id), self.fa)
            # if i >= 4:
            #     break
            # self.process_video(video_url, frames, os.path.join(output_dir, video_id))

        pool.shutdown(wait=True)
        LOG.info(f'Result is saved in {output_dir}.')

    def validate_video_dir(self, video_url, output_dir):
        subvideos = os.listdir(output_dir)
        processed = True
        for subvideo in subvideos:
            path = os.path.join(output_dir, subvideo)
            if not os.path.exists(path + '/landmarks.npy'):
                processed = False
                break

        if processed:
            # Check for images
            for subvideo in subvideos:
                images = glob.glob(os.path.join(output_dir, subvideo, '*.jpg'))
                for img_path in images:
                    img = cv2.imread(img_path)
                    if (img.shape[0] * img.shape[1]) < self.min_image_size * self.min_image_size:
                        # Delete whole subvideo dir including landmarks.npy because it does not make sense anymore.
                        shutil.rmtree(os.path.join(output_dir, subvideo))
                        break
            if len(os.listdir(output_dir)) == 0:
                LOG.info(f'Delete invalid processed video: {video_url}')
                shutil.rmtree(output_dir)
            else:
                LOG.info(f'[video={video_url}] Already downloaded and processed, skipping.')

            return True
        else:
            LOG.info('not processed ??')
            return False

    def process_video(self, video_path, output_dir, fa):
        if os.path.exists(output_dir):
            if self.validate_video_dir(video_path, output_dir):
                self.sem.release()
                return

        vc = cv2.VideoCapture(video_path)
        landmarks = []

        subvideo = 0
        final_output_dir = os.path.join(output_dir, str(subvideo))
        os.makedirs(final_output_dir, exist_ok=True)
        save_frame_num = 0
        first_box = None
        while True:
            ret, frame = vc.read()
            if not ret:
                break

            boxes = common.get_boxes(self.face_driver, frame, threshold=.8)
            if len(boxes) != 1:
                continue
            box = boxes[0]
            if first_box is None:
                first_box = box.copy()

            # check face area
            if (box[2] - box[0]) * (box[3] - box[1]) < self.min_face_size * self.min_face_size:
                continue

            if intersect_area(first_box, box) < 0.3:
                # flush landmarks to final_output_dir.
                if len(landmarks) > self.k:
                    LOG.info(f'Saved {len(landmarks)} frames/landmarks in {final_output_dir}')
                    np.save(os.path.join(final_output_dir, 'landmarks.npy'), np.array(landmarks))
                else:
                    shutil.rmtree(final_output_dir)
                landmarks = []
                first_box = None
                subvideo += 1
                final_output_dir = os.path.join(output_dir, str(subvideo))
                os.makedirs(final_output_dir, exist_ok=True)
                save_frame_num = 0

            # save frame
            file_name = f'{save_frame_num:05d}.jpg'
            cropped_ratio = 0.4
            w = box[2] - box[0]
            h = box[3] - box[1]
            new_box = np.array([
                max(round(box[0] - cropped_ratio * w), 0),
                max(round(box[1] - cropped_ratio * h), 0),
                min(round(box[2] + w * cropped_ratio), frame.shape[1]),
                min(round(box[3] + h * cropped_ratio), frame.shape[0]),
            ]).astype(np.int)
            cropped_frame = frame[new_box[1]:new_box[3], new_box[0]:new_box[2]]
            # get landmarks and accumulate them.
            new_face_box = np.array([
                box[0] - new_box[0],
                box[1] - new_box[1],
                new_box[2] - box[2] + w,
                new_box[3] - box[3] + h,
            ]).astype(int)
            # get landmarks from RGB frame and accumulate them.
            lmark = fa.get_landmarks_from_image(cropped_frame[:, :, ::-1], [new_face_box])
            if len(lmark) == 0:
                continue

            landmarks.append(lmark[0])
            cv2.imwrite(os.path.join(final_output_dir, file_name), cropped_frame)

            save_frame_num += 1

            # cv2.rectangle(
            #     cropped_frame,
            #     (new_face_box[0], new_face_box[1]),
            #     (new_face_box[2], new_face_box[3]),
            #     (0, 250, 0), thickness=1, lineType=cv2.LINE_AA
            # )
            # draw_points(cropped_frame, lmark[0])
            # cv2.imshow('Video', cropped_frame)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     return

        # flush landmarks to final_output_dir.
        if len(landmarks) > self.k:
            np.save(os.path.join(final_output_dir, 'landmarks.npy'), np.array(landmarks))
            LOG.info(f'Saved {len(landmarks)} frames/landmarks in {final_output_dir}')
        else:
            shutil.rmtree(final_output_dir)

        self.sem.release()
        LOG.info(f'End processing video {video_path}')

    def process_text_dir(self, video_dir, video_url, output_dir, fa):
        if os.path.exists(output_dir):
            if self.validate_video_dir(video_url, output_dir):
                self.sem.release()
                return

        txt_paths = glob.glob(os.path.join(video_dir, '*.txt'))
        tmp = tempfile.mktemp(suffix='.mp4')
        try:
            out_path = tmp
            ydl_opts = {
                'format': 'best[height<=480]',
                'outtmpl': out_path,
                'noprogress': True,
            }
            if self.cookie:
                ydl_opts['cookiefile'] = self.cookie

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            vc = cv2.VideoCapture(out_path)
            fps = vc.get(cv2.CAP_PROP_FPS)

            landmarks = []
            prev_frame = None

            subvideo = 0
            for txt_path in txt_paths:
                frames = self.parse_txt(txt_path)

                final_output_dir = os.path.join(output_dir, str(subvideo))
                os.makedirs(final_output_dir, exist_ok=True)
                save_frame_num = 0
                first_box = None

                for data in frames:
                    frame_num, _, _, _, _ = data
                    real_frame_num = round(frame_num / 25 * fps)

                    # read frame by real frame number
                    if prev_frame is not None and real_frame_num > prev_frame:
                        while real_frame_num - prev_frame > 1:
                            vc.grab()
                            prev_frame += 1
                    else:
                        vc.set(cv2.CAP_PROP_POS_FRAMES, real_frame_num)
                    ret, frame = vc.read()
                    if not ret:
                        break

                    boxes = common.get_boxes(self.face_driver, frame, threshold=.9)
                    if len(boxes) != 1:
                        continue
                    box = boxes[0]
                    if first_box is None:
                        first_box = box.copy()

                    # check face area
                    if (box[2] - box[0]) * (box[3] - box[1]) < self.min_face_size * self.min_face_size:
                        continue

                    if intersect_area(first_box, box) < 0.3:
                        # flush landmarks to final_output_dir.
                        if len(landmarks) > self.k:
                            LOG.info(f'Saved {len(landmarks)} frames/landmarks in {final_output_dir}')
                            np.save(os.path.join(final_output_dir, 'landmarks.npy'), np.array(landmarks))
                        else:
                            shutil.rmtree(final_output_dir)
                        landmarks = []
                        first_box = None
                        subvideo += 1
                        final_output_dir = os.path.join(output_dir, str(subvideo))
                        os.makedirs(final_output_dir, exist_ok=True)
                        save_frame_num = 0

                    # save frame
                    file_name = f'{save_frame_num:05d}.jpg'
                    cropped_ratio = 0.4
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    new_box = np.array([
                        max(round(box[0] - cropped_ratio * w), 0),
                        max(round(box[1] - cropped_ratio * h), 0),
                        min(round(box[2] + w * cropped_ratio), frame.shape[1]),
                        min(round(box[3] + h * cropped_ratio), frame.shape[0]),
                    ]).astype(np.int)
                    cropped_frame = frame[new_box[1]:new_box[3], new_box[0]:new_box[2]]
                    # get landmarks and accumulate them.
                    new_face_box = np.array([
                        box[0] - new_box[0],
                        box[1] - new_box[1],
                        new_box[2] - box[2] + w,
                        new_box[3] - box[3] + h,
                    ]).astype(int)
                    # get landmarks from RGB frame and accumulate them.
                    lmark = fa.get_landmarks_from_image(cropped_frame[:, :, ::-1], [new_face_box])
                    if len(lmark) == 0:
                        continue

                    landmarks.append(lmark[0])
                    cv2.imwrite(os.path.join(final_output_dir, file_name), cropped_frame)

                    prev_frame = real_frame_num
                    save_frame_num += 1

                    # cv2.rectangle(
                    #     cropped_frame,
                    #     (new_face_box[0], new_face_box[1]),
                    #     (new_face_box[2], new_face_box[3]),
                    #     (0, 250, 0), thickness=1, lineType=cv2.LINE_AA
                    # )
                    # draw_points(cropped_frame, lmark[0])
                    # cv2.imshow('Video', cropped_frame)
                    # key = cv2.waitKey(0)
                    # if key == 27:
                    #     return

                subvideo += 1
                # flush landmarks to final_output_dir.
                if len(landmarks) > self.k:
                    np.save(os.path.join(final_output_dir, 'landmarks.npy'), np.array(landmarks))
                    LOG.info(f'Saved {len(landmarks)} frames/landmarks in {final_output_dir}')
                else:
                    shutil.rmtree(final_output_dir)

                landmarks = []
        except (youtube_dl.utils.ExtractorError, youtube_dl.utils.DownloadError) as e:
            if '429' in str(e):
                self.sem.release()
                self.stopped = True
                raise
            LOG.info(e)
        except Exception as e:
            LOG.exception(e)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        self.sem.release()

        LOG.info(f'End processing video {video_url}')

    def parse_txt(self, path):
        s = 'FRAME \tX \tY \tW \tH \n'
        with open(path, 'r') as f:
            data = f.read()

        index = data.index(s) + len(s)
        lines = data[index:].split('\n')
        result = []
        for l in lines:
            if len(l) == 0:
                continue
            splitted = l.split()
            if len(splitted) != 5:
                continue
            frame, x, y, w, h = splitted
            result.append((int(frame), int(float(x)), int(float(y)), int(float(w)), int(float(h))))

        return result


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)

    face_driver = driver.load_driver('openvino')().load_model(face_model_path)
    vox = VOXCeleb(args.data_dir, face_driver, args.cookie, workers=args.workers)
    vox.process_videos(args.output)


if __name__ == '__main__':
    main()
