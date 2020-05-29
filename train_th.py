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
    parser.add_argument('--lr', type=float, default=0.0002)
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

    embedder = model_.Embedder(h).build()
    discr = model_.Discriminator(dataset.get_video_num())
    gen = model_.Generator(h)

    # gen([np.random.randn(1, 256, 256, 3).astype(np.float32), np.random.randn(1, 512, 1).astype(np.float32)])

    optimizer_g = tf.keras.optimizers.Adam(args.lr)
    optimizer_d = tf.keras.optimizers.Adam(args.lr)
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    checkpoint = tf.train.Checkpoint(
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        generator=gen,
        discriminator=discr
    )
    checkpoint_man = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint_man.restore_or_initialize()

    # if len(glob.glob(checkpoint_prefix + '*')) > 0:
    #     checkpoint.restore(checkpoint_prefix)
    # LOG.info(model.summary())

    mode = args.mode

    if mode == 'train':
        writer = tf.summary.create_file_writer(os.path.join(args.model_dir))
        # tf.summary.trace_on(graph=True, profiler=True)
        loss_g = loss.LossG(img_size=[h, w, 3])
        scheduler = train.Scheduler(initial_learning_rate=args.lr, epochs=args.epochs)

        # @tf.function
        def step(k_images, k_landmarks, l_image, l_landmark, step_i, finetune=False):
            # TODO: fix video id
            video_id = 0
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # compute average embedding vectors by K frames
                # embs = tf.zeros([args.batch_size * k, 512, 1])
                # for i, (image, lmark) in enumerate(zip(k_images, k_landmarks)):
                #     emb = embedder([tf.expand_dims(image, 0), tf.expand_dims(lmark, 0)])
                #     embs[i] = emb
                embs = embedder([k_images, k_landmarks])
                embs = tf.reshape(embs, [args.batch_size, k, 512, 1])
                embedding = tf.reduce_mean(embs, axis=1)  # out B*512*1

                # Train Generator and Discriminator
                fake_out = gen([l_landmark, embedding])

                score_fake, hat_res_list = discr([fake_out, l_landmark, video_id])
                score_real, res_list = discr([l_image, l_landmark, video_id])

                gen_loss = loss_g(l_image, fake_out, score_fake, res_list, hat_res_list, embs, discr.W_i, video_id)
                # adv_loss = loss_adv(score_fake, res_list, hat_res_list)
                # mch_loss = loss_match()
                loss_dfake = loss.loss_dscfake(score_fake)
                loss_dreal = loss.loss_dscreal(score_real)
                d_loss = loss_dreal + loss_dfake

            eg_variables = gen.trainable_variables + embedder.trainable_variables
            discr_variables = discr.trainable_variables

            # Some variables won't be trained in not-finetuning mode
            def delete_unneeded_variables(variables):
                i = 0
                while i < len(variables):
                    if variables[i].name in {'generator/psi:0', 'discriminator/w_prime:0'}:
                        del variables[i]
                    i += 1
                return variables

            if not finetune:
                delete_unneeded_variables(eg_variables)
                delete_unneeded_variables(discr_variables)

            gradients_of_generator = gen_tape.gradient(gen_loss, eg_variables)
            gradients_of_discriminator = disc_tape.gradient(d_loss, discr_variables)

            optimizer_g.apply_gradients(zip(gradients_of_generator, eg_variables))
            optimizer_d.apply_gradients(zip(gradients_of_discriminator, discr_variables))

            if step_i % 5 == 0:
                with writer.as_default():
                    tf.summary.scalar('gen_loss', gen_loss, step=step_i)
                    tf.summary.scalar('disc_loss', d_loss, step=step_i)
                print(f'Step {step_i}, gen_loss={gen_loss}, discr_loss={d_loss}')

        refer_images, test_landmarks, test_images = dataset.get_test_batch(k)
        test_landmark, test_result = np.expand_dims(test_landmarks[0], 0), np.expand_dims(test_images[0], 0)

        for epoch in range(args.epochs):
            input_fn = dataset.get_input_fn()
            for step_i, ((images, landmarks), labels) in enumerate(input_fn):
                k_images = labels
                k_landmarks = landmarks
                l_image = tf.expand_dims(random.choice(k_images), 0)
                l_landmark = tf.expand_dims(random.choice(k_landmarks), 0)
                step(k_images, k_landmarks, l_image, l_landmark, step_i)

                # Save the model every epoch (for now)
            if (epoch + 1) % 1 == 0:
                checkpoint_man.save(checkpoint_number=epoch)

            embs = embedder([test_images, test_landmarks])
            embs = tf.reshape(embs, [args.batch_size, k, 512, 1])
            embedding = tf.reduce_mean(embs, axis=1)  # out B*512*1
            test_pred = gen([test_landmark, embedding])
            with writer.as_default():
                tf.summary.scalar('val_loss', tf.reduce_mean(tf.abs(test_result - test_pred)))
                LOG.info(tf.summary.image("Result", test_pred, step=epoch))

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
