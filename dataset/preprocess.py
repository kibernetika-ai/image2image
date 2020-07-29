import argparse
import glob
import queue
import os
import sys
import threading
import time

import cv2
import face_alignment
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir')
parser.add_argument('--output')
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--start-percent', type=float, default=0.0)

args = parser.parse_args()

path_to_mp4 = args.data_dir
device = torch.device('cuda:0')
saves_dir = args.output

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)


def print_fun(s):
    print(s)
    sys.stdout.flush()


def generate_landmarks(frame, face_aligner):
    input = frame
    preds = face_aligner.get_landmarks(input)[0]

    return preds


def process_images(video_dir, lm_queue: queue.Queue, out_dir):
    videos = sorted([os.path.join(video_dir, v) for v in os.listdir(video_dir)])

    # First check
    new_video_dir = get_new_video_dir(os.path.join(video_dir, 'dummy'), out_dir)
    if os.path.exists(os.path.join(new_video_dir, 'landmarks.npy')):
        jpgs = glob.glob(new_video_dir + '/*.jpg')
        lm = np.load(new_video_dir + '/landmarks.npy')
        if len(lm) == len(jpgs):
            print_fun(f'Skip already processed {video_dir}...')
            return
        else:
            print_fun(f'Re-process {video_dir}...')

    for video_i, video_path in enumerate(videos):
        print_fun(f'Process {video_path}...')
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
            last = frame_id == frame_num and video_i == len(videos) - 1
            lm_queue.put((rgb, video_path, last))

        cap.release()


def get_new_video_dir(video_path, output_dir):
    splitted = video_path.split('/')
    video_id = splitted[-2]
    person_id = splitted[-3]
    new_dir = os.path.join(output_dir, person_id, video_id)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


class LandmarksQueue(object):
    def __init__(self, q: queue.Queue, root_dir, threads=1):
        self.landmarks = []
        self.q = q
        self.root_dir = root_dir
        self.save_q = queue.Queue(maxsize=q.maxsize)
        self.lm = queue.Queue(maxsize=q.maxsize)
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0')
        # Dry run
        print_fun('Face alignment dry run...')
        self.face_aligner.get_landmarks(np.random.randint(0, 255, size=(256, 256, 3)))
        self.face_aligner.face_alignment_net(
            torch.from_numpy(np.random.randint(0, 255, size=(1, 3, 256, 256))).float().div(255.).to(torch.device('cuda'))
        )[-1].detach().cpu()
        print_fun('Done.')

        self.save_frame_id = 0
        self.lock = threading.Lock()
        self.num_threads = threads
        self.threads = []

    def start_process(self):
        for i in range(self.num_threads):
            t = threading.Thread(target=self.process_lm, daemon=True)
            t.start()
            self.threads.append(t)
        t = threading.Thread(target=self.process_save, daemon=True)
        t.start()
        self.threads.append(t)
        t = threading.Thread(target=self.save_landmarks, daemon=True)
        t.start()
        self.threads.append(t)

    def save_landmarks(self):
        while True:
            item = self.lm.get()
            if isinstance(item, str):
                if item == 'stop':
                    break
            else:
                lmarks = item[0]
                save_path = item[1]
                print_fun(f'Processed {len(lmarks)} landmarks: {time.time() - self.start}')
                self.start = time.time()

                print_fun(f'save {save_path}')
                np.save(save_path, lmarks)

    def process_lm(self):
        self.start = time.time()
        while True:
            item = self.q.get()
            if isinstance(item, str):
                if item == 'stop':
                    break
            else:
                frame = item[0]
                video_dir = get_new_video_dir(item[1], self.root_dir)
                last = item[2]
                try:
                    landmark = generate_landmarks(frame, self.face_aligner)
                except Exception as e:
                    print_fun(e)
                    continue

                with self.lock:
                    cropped_frame, recomputed_landmark = self.crop_landmark(frame, landmark)
                    self.landmarks.append(recomputed_landmark)
                    self.save_q.put((cropped_frame, video_dir, self.save_frame_id))
                    self.save_frame_id += 1
                    if last:
                        save_path = os.path.join(video_dir, 'landmarks.npy')
                        self.lm.put((np.stack(self.landmarks).copy(), save_path))
                        self.landmarks = []
                        self.save_frame_id = 0

    @staticmethod
    def crop_landmark(frame, landmark):
        # crop frame
        maxx, maxy = np.max(landmark, axis=0)
        minx, miny = np.min(landmark, axis=0)
        margin = 0.4
        margin_top = margin + 0.3
        miny = max(int(miny - (maxy - miny) * margin_top), 0)
        maxy = min(int(maxy + (maxy - miny) * margin), frame.shape[0])
        minx = max(int(minx - (maxx - minx) * margin), 0)
        maxx = min(int(maxx + (maxx - minx) * margin), frame.shape[1])
        new_frame = frame[miny:maxy, minx:maxx]
        new_landmark = landmark.copy()
        new_landmark -= [minx, miny]

        return new_frame, new_landmark

    def process_save(self):
        while True:
            item = self.save_q.get()
            if isinstance(item, str):
                if item == 'stop':
                    break
            else:
                bgr = cv2.cvtColor(item[0], cv2.COLOR_RGB2BGR)
                video_dir = item[1]
                frame_id = item[2]
                cv2.imwrite(os.path.join(video_dir, f'{frame_id:05d}.jpg'), bgr)

    def stop(self):
        for i in range(self.num_threads):
            self.q.put('stop')
        while not self.q.empty():
            time.sleep(0.1)

        self.save_q.put('stop')
        self.lm.put('stop')
        for t in self.threads:
            t.join(10)


video_paths = glob.glob(os.path.join(path_to_mp4, '**/*'))
if args.reverse:
    video_paths.reverse()

start_index = 0
if args.start_percent > 0:
    start_index = int(len(video_paths) * args.start_percent)

lm_queue = queue.Queue(maxsize=300)
landmarks_queue = LandmarksQueue(lm_queue, args.output, threads=args.threads)
landmarks_queue.start_process()

print_fun(f'Number of videos: {len(video_paths)}')
for i, video_dir in enumerate(video_paths):
    if i < start_index:
        continue
    print_fun(f'[{i}/{len(video_paths)}] Process dir {video_dir}')
    process_images(video_dir, lm_queue, args.output)

print_fun('Done.')
print_fun('Waiting stop threads...')
landmarks_queue.stop()
print_fun('Done.')
