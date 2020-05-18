import argparse
import glob
import json
import os
import tempfile

from concurrent import futures
import cv2
import youtube_dl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--output', default='output')

    return parser.parse_args()


class VOXCeleb(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # structure: {root}/txt/{idXXXX}/{video_id}/{XXXX}.txt

    def process_videos(self, output_dir):
        videos = sorted(glob.glob(os.path.join(self.data_dir, '*/*/*')))
        pool = futures.ThreadPoolExecutor(max_workers=4)
        for video_dir in videos:
            video_id = video_dir.split('/')[-1]
            video_url = f'https://www.youtube.com/watch?v={video_id}'

            pool.submit(self.process_video, video_dir,  video_url, os.path.join(output_dir, video_id))
            # self.process_video(video_url, frames, os.path.join(output_dir, video_id))

        pool.shutdown(wait=True)

        print(f'Result is saved in {output_dir}.')

    def process_video(self, video_dir, video_url, output_dir):
        print(f'Start processing video {video_url}...')
        # if os.path.exists(output_dir):
        #     files = os.listdir(output_dir)
        #     if len(files) == len(frames) + 1:
        #         print(f'[video={video_url}] Already downloaded and processed, skipping.')
        #         return

        txt_paths = glob.glob(os.path.join(video_dir, '*.txt'))
        tmp = tempfile.mktemp(suffix='.mp4')
        try:
            out_path = tmp
            ydl_opts = {
                'format': 'best[height<=480]',
                'outtmpl': out_path,
                'noprogress': True,
            }
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url)

            vc = cv2.VideoCapture(out_path)
            fps = vc.get(cv2.CAP_PROP_FPS)

            boxes = {}
            boxes_file = os.path.join(output_dir, 'boxes.json')
            prev_frame = None

            for txt_path in txt_paths:
                frames = self.parse_txt(txt_path)
                for data in frames:
                    frame_num, x, y, w, h = data
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

                    # save frame
                    file_name = f'{real_frame_num:05d}.jpg'
                    cropped_ratio = 0.2
                    new_box = [
                        max(round(x - cropped_ratio * w), 0),
                        max(round(y - cropped_ratio * h), 0),
                        min(round(x + w * (1.0 + cropped_ratio)), frame.shape[1]),
                        min(round(y + h * (1.0 + cropped_ratio)), frame.shape[0]),
                    ]
                    cropped_frame = frame[new_box[1]:new_box[3], new_box[0]:new_box[2]]
                    cv2.imwrite(os.path.join(output_dir, file_name), cropped_frame)
                    new_box = [
                        max(data[1] - new_box[0], 0),
                        max(data[2] - new_box[1], 0),
                        min(new_box[2] - data[1], cropped_frame.shape[1]),
                        min(new_box[3] - data[2], cropped_frame.shape[0]),
                    ]
                    # Write in format x1,y1,x2,y2
                    if file_name in boxes:
                        boxes[file_name].append(new_box)
                    else:
                        boxes[file_name] = [new_box]
                    prev_frame = real_frame_num
                    # cv2.rectangle(
                    #     cropped_frame,
                    #     (new_box[0], new_box[1]),
                    #     (new_box[2], new_box[3]),
                    #     (0, 250, 0), thickness=1, lineType=cv2.LINE_AA
                    # )
                    # cv2.imshow('Video', cropped_frame)
                    # key = cv2.waitKey(0)
                    # if key == 27:
                    #     return

                print(f'Saved {len(frames)} frames in {output_dir}')

            with open(boxes_file, 'w') as f:
                f.write(json.dumps(boxes, indent=2))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        print(f'End processing video {video_url}')

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
            result.append((int(frame), int(x), int(y), int(w), int(h)))

        return result


def main():
    args = parse_args()
    vox = VOXCeleb(args.data_dir)
    vox.process_videos(args.output)


if __name__ == '__main__':
    main()
