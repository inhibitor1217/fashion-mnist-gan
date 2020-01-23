import numpy as np
import math
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Lambda, BatchNormalization, Activation
from tensorflow.keras import Input, Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def normalize_image(img):
    return img / 127.5 - 1.

def load_fashion_mnist(single_class=-1):
    (train_x, train_y), _ = fashion_mnist.load_data()
    train_x = np.expand_dims(train_x, axis=-1)
    if single_class != -1:
        return normalize_image(train_x[train_y == single_class])
    else:
        return normalize_image(train_x)

def format_image(x):
    if len(x.shape) == 4:
        (num_samples, img_height, img_width, _) = x.shape
        rows = math.ceil(num_samples / 16)
        cols = min(num_samples, 16)
        out  = np.zeros(shape=(img_height * rows, img_width * cols))
        for row in range(rows):
            for col in range(cols):
                out[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width] = x[row*cols+col, :, :, 0]
        return out
    elif len(x.shape) == 3:
        return x[:, :, 0]
    else:
        return x

def imshow(x):
    plt.imshow(format_image(x))

IMAGE_INPUT_SHAPE = (28, 28, 1)
NOISE_INPUT_SHAPE = (1, 1, 128)

def define_generator():
    x = _input = Input(shape=NOISE_INPUT_SHAPE) # noise

    x = Conv2DTranspose(64, 3, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(16, 5, 1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(8, 2, 2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(8, 2, 2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(1, 3, 1, padding='same', activation='tanh')(x)

    model = Model(_input, x)

    return model

def define_discriminator():
    x = _input = Input(shape=IMAGE_INPUT_SHAPE)

    x = Conv2D(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, 4, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, 2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dense(1, activation='linear')(x)

    model = Model(_input, x)
    return model

def sample_images(num_samples=256):
    noise = np.random.normal(size=(num_samples, *NOISE_INPUT_SHAPE))
    fakes = g.predict(noise)
    imshow(fakes)

if __name__ == '__main__':
    g = define_generator()
    d = define_discriminator()

    _input_image = Input(shape=IMAGE_INPUT_SHAPE)
    _input_noise = Input(shape=NOISE_INPUT_SHAPE)

    real_prediction = d(_input_image)
    fake_prediction = d(g(_input_noise))

    def relativistic_average(args):
        pred0, pred1 = args
        return pred0 - K.mean(pred1, axis=0) # average over batches

    real_to_fake = Lambda(relativistic_average)([real_prediction, fake_prediction])
    fake_to_real = Lambda(relativistic_average)([fake_prediction, real_prediction])
    real_to_fake_sigmoid = Activation('sigmoid')(real_to_fake)
    fake_to_real_sigmoid = Activation('sigmoid')(fake_to_real)

    generator_model = Model([_input_image, _input_noise], [real_to_fake_sigmoid, fake_to_real_sigmoid])
    g.trainable = True
    d.trainable = False
    generator_model.compile(optimizer=Adam(lr=1e-4, beta_1=.9, clipvalue=1, clipnorm=1), loss='binary_crossentropy')

    discriminator_model = Model([_input_image, _input_noise], [real_to_fake_sigmoid, fake_to_real_sigmoid])
    g.trainable = False
    d.trainable = True
    discriminator_model.compile(optimizer=Adam(lr=1e-4, beta_1=.9, clipvalue=1, clipnorm=1), loss='binary_crossentropy')

    d_losses = []
    g_losses = []

    epochs = 100
    x_train = load_fashion_mnist()
    num_train = x_train.shape[0]
    batch_size = min(1024, num_train)
    steps = num_train // batch_size

    for epoch in range(epochs):
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        for step in range(steps):
            noise = np.random.normal(0, 1, size=(batch_size, *NOISE_INPUT_SHAPE))
            image = x_train[step * batch_size:(step + 1) * batch_size]

            y_real_smoothed = np.random.uniform(0.9, 1.0, size=(batch_size, 1))
            y_fake_smoothed = np.random.uniform(0.0, 0.1, size=(batch_size, 1))
            y_real = np.ones(shape=(batch_size, 1))
            y_fake = np.zeros(shape=(batch_size, 1))

            g.trainable = False
            d.trainable = True
            d_loss = discriminator_model.train_on_batch([image, noise], [y_real_smoothed, y_fake_smoothed])[0]

            g.trainable = True
            d.trainable = False
            g_loss = generator_model.train_on_batch([image, noise], [y_fake_smoothed, y_real_smoothed])[0]

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            print(f'\repoch={epoch}, step={step}, d_loss={d_loss}, g_loss={g_loss}')

    plt.figure(figsize=(8, 8))
    plt.plot(d_losses)
    plt.plot(g_losses)
    plt.legend(['d_losses', 'g_losses'])
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.gray()
    sample_images()
    plt.colorbar()