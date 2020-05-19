import tensorflow as tf
from tensorflow.keras import layers


def build_model(image_shape=(320, 320)):
    # Build U-Net model
    picture = layers.Input((image_shape[0], image_shape[1], 3))
    landmarks = layers.Input((68, 2))

    l1 = layers.Dense(3, use_bias=False)(landmarks)
    l1 = layers.BatchNormalization()(l1)
    l1 = layers.LeakyReLU()(l1)
    # split on 4 groups
    l2 = layers.Reshape([4, 17, 3])(l1)

    l2 = layers.Conv2DTranspose(
        64, (2, 2), strides=(image_shape[0] // 4, 1), padding='same'
    )(l2)
    l2 = layers.Conv2DTranspose(
        32, (3, 3), strides=(1, image_shape[1] // 17 + 1), padding='same', output_padding=(0, 15)
    )(l2)
    l2 = layers.BatchNormalization()(l2)
    l2 = layers.Dropout(0.1)(l2)
    # s = layers.Lambda(lambda x: x / 255)(inputs)

    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(picture)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    # outputs = layers.Dense(3, activation='sigmoid', kernel_initializer='he_normal')(c9)
    # outputs = layers.Conv2D(3, (1, 1), strides=(1, 1), activation='sigmoid')(c9)
    pre_outputs = layers.concatenate([c9, l2])
    outputs = layers.Conv2D(3, (1, 1), strides=(1, 1), activation='sigmoid')(pre_outputs)
    model = tf.keras.Model(inputs=[picture, landmarks], outputs=[outputs])
    return model
