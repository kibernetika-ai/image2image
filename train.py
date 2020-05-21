import argparse
import glob
import json
import logging
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
import tqdm

import common
import model_def


LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', action='store_true')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--model-dir', default='train')
    parser.add_argument('--output', default='saved_model')
    parser.add_argument('--data-dir')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resolution', default="256x256")
    parser.add_argument('--loss', default="mse")

    return parser.parse_args()


def parse_resolution(res):
    splitted = res.split('x')
    if len(splitted) != 2:
        raise RuntimeError("Resolution must be in form WxH")

    return int(splitted[0]), int(splitted[1])


class ImageDataset:
    def __init__(self, data_dir, batch_size=1,
                 img_num=10, width=256, height=256, shuffle=True, val_split=0.0):
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
                # load boxes.json
                # with open(os.path.join(video_dir, 'boxes.json')) as f:
                #     boxes = json.loads(f.read())
                # load landmarks.json
                with open(os.path.join(video_dir, 'landmarks.json')) as f:
                    landmarks = json.loads(f.read())

                img_paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
                reference = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
                reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
                reference = cv2.resize(reference, (self.width, self.resize_height))
                reference = common.normalize(reference)

                img_paths = img_paths[1:]
                if self.shuffle:
                    random.shuffle(img_paths)

                for img_path in img_paths:
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.width, self.resize_height))

                    # Normalization
                    img = common.normalize(img)
                    basename = os.path.basename(img_path)
                    landmark = np.array(landmarks[basename]).astype(np.float32)
                    landmark_img = common.landmarks_to_img(landmark, img.shape)

                    yield (reference, landmark_img), img
        return generate_batches

    def _get_ds_from_list(self, dir_list):
        dataset = tf.data.Dataset.from_generator(
            self.get_generator(dir_list),
            ((tf.float32, tf.float32), tf.float32),
            (
                (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                tf.TensorShape([None, None, 3])
            )
        )
        return dataset.batch(self.batch_size).prefetch(self.batch_size * 2)

    def get_input_fn(self):
        return self._get_ds_from_list(self.train_dirs)

    def get_test_batch(self):
        gen = self.get_generator(self.train_dirs)
        for i in gen():
            return i

    def get_val_input_fn(self):
        return self._get_ds_from_list(self.val_dirs)


mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


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
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
    w, h = parse_resolution(args.resolution)
    dataset = ImageDataset(args.data_dir, args.batch_size, width=w, height=h)

    # inp = dataset.get_input_fn()
    # it = inp.as_numpy_iterator()
    # for i in it:
    #     (_, landmarks), img = i
    #     l_img = landmarks[0]
    #     l_img = (l_img * 255.0).astype(np.uint8).clip(0, 255)
    #     cv2.imshow('img', l_img)
    #     k = cv2.waitKey(0)
    #     if k == 27:
    #         break
    # return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        LOG.info("=" * 50)
        LOG.info(f"Set memory growth to {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
        LOG.info("=" * 50)

    model = model_def.build_model(image_shape=(h, w))
    # LOG.info(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        # loss=[CrossEntropyLoss(num_classes=dataset.num_classes()), None],
        loss=[args.loss],
        metrics=['accuracy']
    )
    mode = args.mode

    if mode == 'train':
        scheduler = Scheduler(initial_learning_rate=args.lr, epochs=args.epochs)
        file_writer_cm = tf.summary.create_file_writer(args.model_dir)
        (test_image, test_landmark), test_result = dataset.get_test_batch()
        test_image = np.expand_dims(test_image, axis=0)
        test_landmark = np.expand_dims(test_landmark, axis=0)

        def log_image(batch, logs):
            # Use the model to predict the values from the validation dataset.
            if batch % 100 != 0:
                return

            test_pred = model.predict_on_batch((test_image, test_landmark))

            # Log the confusion matrix as an image summary.
            with file_writer_cm.as_default():
                tf.summary.image("Result", test_pred, step=batch)

        callbacks = [
            tf.keras.callbacks.LambdaCallback(on_batch_end=log_image),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.model_dir),
                update_freq=50, write_images=True
            ),
        ]
        if args.lr_scheduler:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler.schedule, verbose=1))
        model.fit(
            x=dataset.get_input_fn(),
            # validation_data=dataset.get_val_input_fn(),
            # batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1 if sys.stdout.isatty() else 2,
            callbacks=callbacks,
            # tf.keras.callbacks.ModelCheckpoint(
            #     os.path.join(args.model_dir, 'checkpoint'),
            #     verbose=1,
            # ),
        )
        model.save(os.path.join(args.model_dir, 'checkpoint'), save_format='tf')
        LOG.info(f'Checkpoint is saved to {os.path.join(args.model_dir, "checkpoint")}.')

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
        LOG.info(f'Saved to {args.output}')

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
    #     LOG.info(f'Saved to {saved_path}.')


if __name__ == '__main__':
    main()
