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
from model_gan import model as model_
from model_gan import loss

import train


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
    parser.add_argument('--finetune', action='store_true')

    return parser.parse_args()


mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
    w, h = train.parse_resolution(args.resolution)
    k = 8
    dataset = train.ImageDataset(args.data_dir, args.batch_size * k, width=w, height=h)

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

    gen = model_.Generator(h)
    discr = model_.Discriminator()
    embedder = model_.Embedder(h)
    checkpoint_dir_g = os.path.join(args.model_dir, 'checkpoint_g')
    checkpoint_dir_d = os.path.join(args.model_dir, 'checkpoint_d')
    checkpoint_dir_e = os.path.join(args.model_dir, 'checkpoint_e')
    if os.path.exists(checkpoint_dir_g):
        gen.load_weights(checkpoint_dir_g)

    if os.path.exists(checkpoint_dir_d):
        gen.load_weights(checkpoint_dir_d)

    if os.path.exists(checkpoint_dir_e):
        gen.load_weights(checkpoint_dir_e)
    # LOG.info(model.summary())

    mode = args.mode

    if mode == 'train':
        loss_g = loss.LossG()
        loss_d = loss.loss_dsc
        scheduler = train.Scheduler(initial_learning_rate=args.lr, epochs=args.epochs)
        optimizer_g = tf.keras.optimizers.Adam(args.lr)
        optimizer_d = tf.keras.optimizers.Adam(args.lr)

        @tf.function
        def step(k_images, k_landmarks, l_image, l_landmark):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # compute average embedding vectors by K frames
                embs = np.zeros([args.batch_size * k, 512, 1])
                for i, (image, lmark) in enumerate(zip(k_images, k_landmarks)):
                    emb = embedder(image, lmark)
                    embs[i] = emb
                embs = tf.reshape(embs, [args.batch_size, k, 512, 1])
                e_hat = tf.reduce_mean(embs, axis=1)

                x_hat = gen(l_landmark, e_hat)

                real_output = discr(image, training=True)
                fake_output = discr(generated_images, training=True)

                gen_loss = loss_g(fake_output)
                disc_loss = loss_d(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discr.trainable_variables)

            optimizer_g.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
            optimizer_d.apply_gradients(zip(gradients_of_discriminator, discr.trainable_variables))

        (test_image, test_landmark), test_result = dataset.get_test_batch()
        test_image = np.expand_dims(test_image, axis=0)
        test_landmark = np.expand_dims(test_landmark, axis=0)

        for epoch in range(args.epochs):
            input_fn = dataset.get_input_fn()
            for images, landmarks, labels in input_fn():
                k_images = labels[:k]
                k_landmarks = landmarks[:k]
                l_image = labels[-1]
                l_landmark = landmarks[-1]
                step(k_images, k_landmarks, l_image, l_landmark)

            # Save the model every 15 epochs
            if (epoch + 1) % 5 == 0:
                gen.save(checkpoint_dir_e)
                discr.save(checkpoint_dir_e)
                embedder.save(checkpoint_dir_e)
        # model.save(os.path.join(args.model_dir, 'checkpoint'), save_format='tf', include_optimizer=False)
        LOG.info(f'Checkpoint is saved to {os.path.join(args.model_dir, "checkpoint")}.')

    if mode == 'validate':
        pass
        # m = tf.keras.models.load_model(os.path.join(args.model_dir, 'checkpoint'))
        # model.load_weights()
        # gen = dataset.get_generator(dataset.val_dirs)()
        # for i, (imgs, label) in tqdm.tqdm(enumerate(gen)):
        #     output = m.predict_on_batch(np.expand_dims(imgs, axis=0))
        #     val_output = (output[0] * 255.0).astype(np.uint8).clip(0, 255)
        #     cv2.imwrite(f'{i}.png', val_output[:, :, ::-1])

    if mode == 'export' or args.export:
        # model.load_weights(os.path.join(args.model_dir, 'checkpoint'))
        # model.outputs = model.outputs[::-1]
        # model.output_names = model.output_names[::-1]
        # model.save(args.output, save_format='tf', include_optimizer=False)
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
