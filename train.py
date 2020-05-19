import argparse
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
import tqdm

import model_def


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--model-dir', default='train')
    parser.add_argument('--output', default='saved_model')
    parser.add_argument('--data-dir')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resolution', default="1280x720")
    parser.add_argument('--loss', default="mae")

    return parser.parse_args()


def parse_resolution(res):
    splitted = res.split('x')
    if len(splitted) != 2:
        raise RuntimeError("Resolution must be in form WxH")

    return int(splitted[0]), int(splitted[1])


class VideoDataset:
    def __init__(self, data_dir, batch_size=1,
                 img_num=10, width=1280, height=720, shuffle=True, val_split=0.1):
        self.batch_size = batch_size

        # structure: {root}/{video_id}/{XXXXX}.jpg
        # structure: {root}/{video_id}/boxes.json
        data_dir = data_dir.rstrip('/')
        self.video_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)]
        if val_split > 0:
            self.train_dirs = self.video_dirs[:int(len(self.video_dirs) * (1 - val_split))]
            self.val_dirs = self.video_dirs[int(len(self.video_dirs) * (1 - val_split)):]
        else:
            self.train_dirs = self.video_dirs
            self.val_dirs = []

        self.shuffle = shuffle
        self.img_num = img_num
        self.width = width
        self.resize_height = height
        self.height = height // 16 * 16

    def get_generator(self, dir_list):
        target_list = dir_list

        def generate_batches():
            if self.shuffle:
                random.shuffle(target_list)

            for video_dir in target_list:
                start, end = 1, self.img_num
                image_batch = []
                while True:
                    lossless_path = os.path.join(video_dir, f'lossless-{end}.png')
                    if not os.path.exists(lossless_path):
                        break
                    lossless_img = cv2.imread(lossless_path)
                    lossless_img = cv2.cvtColor(lossless_img, cv2.COLOR_BGR2RGB)
                    lossless_img = cv2.resize(lossless_img, (self.width, self.resize_height))
                    for i in range(start, end+1):
                        path = os.path.join(video_dir, f'{i:02d}.jpg')
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.width, self.resize_height))
                        image_batch.append(img)

                    imgs = np.concatenate(image_batch, axis=-1)
                    # imgs = np.stack(image_batch)

                    # Normalization
                    imgs = normalize(imgs)
                    lossless_img = normalize(lossless_img)

                    yield imgs, lossless_img
                    start += self.img_num
                    end += self.img_num
                    image_batch = []
        return generate_batches

    def _get_ds_from_list(self, dir_list):
        dataset = tf.data.Dataset.from_generator(
            self.get_generator(dir_list),
            (tf.float32, tf.float32),
            (
                tf.TensorShape([self.height, self.width, 3 * self.img_num]),
                tf.TensorShape([self.height, self.width, 3])
            )
        )
        return dataset.batch(self.batch_size).prefetch(self.batch_size * 2)

    def get_input_fn(self):
        return self._get_ds_from_list(self.train_dirs)

    def get_val_input_fn(self):
        return self._get_ds_from_list(self.val_dirs)


mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


def normalize(img):
    rank = len(img.shape)
    height_dim = 1 if rank == 4 else 0
    nearest_multiple_16 = img.shape[height_dim] // 16 * 16
    if nearest_multiple_16 != img.shape[height_dim]:
        # crop by height
        crop_need = img.shape[height_dim] - nearest_multiple_16
        if rank == 4:
            img = img[:, crop_need // 2:-crop_need // 2, :, :]
        else:
            img = img[crop_need // 2:-crop_need // 2, :, :]

    return img.astype(np.float32) / 255.0


class Scheduler:
    def __init__(self, initial_learning_rate=0.05, epochs=10):
        self.epochs = epochs
        self.learning_rate = initial_learning_rate

    def schedule(self, epoch):
        if epoch < 0.3 * self.epochs:
            return self.learning_rate
        elif epoch < 0.5 * self.epochs:
            return self.learning_rate / 5
        elif epoch < 0.75 * self.epochs:
            return self.learning_rate / 25
        else:
            return self.learning_rate / 250


def main():
    args = parse_args()
    w, h = parse_resolution(args.resolution)
    dataset = VideoDataset(args.data_dir, args.batch_size, width=w, height=h)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print("=" * 50)
        print(f"Set memory growth to {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
        print("=" * 50)

    model = model_def.build_model()
    # print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        # loss=[CrossEntropyLoss(num_classes=dataset.num_classes()), None],
        loss=[args.loss],
        metrics=['accuracy']
    )
    mode = args.mode

    if mode == 'train':
        scheduler = Scheduler(initial_learning_rate=args.lr, epochs=args.epochs)
        model.fit(
            x=dataset.get_input_fn(),
            validation_data=dataset.get_val_input_fn(),
            # batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1 if sys.stdout.isatty() else 2,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(scheduler.schedule, verbose=1),
                tf.keras.callbacks.TensorBoard(log_dir=args.model_dir, update_freq=30),
                # tf.keras.callbacks.ModelCheckpoint(
                #     os.path.join(args.model_dir, 'checkpoint'),
                #     verbose=1,
                # ),
            ]
        )
        # validation.evaluate(
        #     model,
        #     dataset.get_query_input_fn(),
        #     dataset.get_test_input_fn(),
        #     dist_metric='euclidean',
        # )
        model.save(os.path.join(args.model_dir, 'checkpoint'), save_format='tf')
        print(f'Checkpoint is saved to {os.path.join(args.model_dir, "checkpoint")}.')

    if mode == 'validate':
        m = tf.keras.models.load_model(os.path.join(args.model_dir, 'checkpoint'))
        # model.load_weights()
        gen = dataset.get_generator(dataset.val_dirs)()
        for i, (imgs, label) in tqdm.tqdm(enumerate(gen)):
            output = m.predict_on_batch(np.expand_dims(imgs, axis=0))
            val_output = (output[0] * 255.0).astype(np.uint8).clip(0, 255)
            cv2.imwrite(f'{i}.png', val_output[:, :, ::-1])

    if mode == 'export' or args.export:
        model.load_weights(os.path.join(args.model_dir, 'checkpoint'))
        model.outputs = model.outputs[::-1]
        model.output_names = model.output_names[::-1]
        model.save(args.output, save_format='tf')
        print(f'Saved to {args.output}')

    # config_proto = tf.compat.v1.ConfigProto(log_device_placement=True)
    # config = tf.estimator.RunConfig(
    #     model_dir=args.model_dir,
    #     save_summary_steps=100,
    #     keep_checkpoint_max=5,
    #     log_step_count_steps=10,
    #     session_config=config_proto,
    # )
    # estimator_model = tf.keras.estimator.model_to_estimator(
    #     keras_model=model,
    #     model_dir=args.model_dir,
    #     config=config,
    #     checkpoint_format='saver',
    # )
    #
    # if mode == 'train':
    #     estimator_model.train(
    #         input_fn=dataset.get_input_fn,
    #         steps=args.steps,
    #     )
    # elif mode == 'export':
    #     saved_path = estimator_model.export_saved_model(
    #         args.model_dir,
    #     )
    #     print(f'Saved to {saved_path}.')


if __name__ == '__main__':
    main()
