import glob
import time

import torch
from torch.utils.data import Dataset

from .video_extraction_conversion import *


class PreprocessDataset(Dataset):
    def __init__(self, path_to_preprocess, k=1, frame_shape=224):
        self.k = k
        self.path_to_preprocess = path_to_preprocess
        self.frame_shape = frame_shape

        self.video_dirs = glob.glob(os.path.join(path_to_preprocess, '*/*'))
        self.mean_landmark = None

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        np.random.seed(int(time.time()))

        vid_idx = idx
        video_dir = self.video_dirs[vid_idx]
        lm_path = os.path.join(video_dir, 'landmarks.npy')
        jpg_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        # if not jpg_paths:
        #     raise RuntimeError('Dataset does not contain .jpg files.')
        if os.path.exists(lm_path):
            all_landmarks = np.load(lm_path)

        while not os.path.exists(lm_path) or len(all_landmarks) != len(jpg_paths):
            vid_idx = vid_idx // 2
            video_dir = self.video_dirs[vid_idx]
            lm_path = os.path.join(video_dir, 'landmarks.npy')
            if not os.path.exists(lm_path):
                continue
            jpg_paths = glob.glob(os.path.join(video_dir, '*.jpg'))
            all_landmarks = np.load(lm_path)
            if len(all_landmarks) != len(jpg_paths):
                continue

        # Select K paths
        random_indices = np.random.randint(0, len(jpg_paths), size=(self.k + 1,))
        paths = np.array(jpg_paths)[random_indices]
        landmarks = all_landmarks[random_indices]

        frames = []
        marks = []
        for i, path in enumerate(paths):
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cur_landmark = landmarks[i].copy()
            if frame.shape[:2] != (self.frame_shape, self.frame_shape):
                x_factor, y_factor = frame.shape[1] / self.frame_shape, frame.shape[0] / self.frame_shape
                frame = cv2.resize(frame, (self.frame_shape, self.frame_shape), interpolation=cv2.INTER_AREA)
                cur_landmark /= [x_factor, y_factor]
            lmark = draw_landmark(cur_landmark, size=frame.shape)
            # cv2.imshow('img', np.hstack((frame, lmark))[:, :, ::-1])
            # cv2.waitKey(0)
            # exit()
            frames.append(frame)
            marks.append(lmark)

        frames = torch.from_numpy(np.array(frames)).type(dtype=torch.float)  # K,256,256,3
        marks = torch.from_numpy(np.array(marks)).type(dtype=torch.float)  # K,256,256,3
        frames = (frames.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,256,256
        marks = (marks.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,256,256
        # frame_mark = frame_mark.requires_grad_(False)

        img = frames[-1]
        mark = marks[-1]
        frames = frames[:self.k]
        marks = marks[:self.k]

        return frames, marks, img, mark, vid_idx


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
