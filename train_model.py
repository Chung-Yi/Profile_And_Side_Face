import cv2
import os
import glob
import numpy as np
import keras
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from operator import itemgetter
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Input, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint, EarlyStopping
from inception_block import VGG16_Inception
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import efficientnet.keras as efn

IMG_PATH = 'images/train/**'
TRAIN_VAL_SPLIT = 0.7
BATCH_SIZE = 32
EPOCHS = 2000
IMG_SIZE = 100

parser = ArgumentParser()
parser.add_argument("model", help="choose a model")
parser.add_argument('--gpu', help="enable gpu", action='store_true')


def get_data_generator(image_list, indices, batch_size=64):
    # images, labels, names = [], [], []

    while True:

        batches = int(len(indices) / batch_size)
        # remainder_samples = len(indices) % batch_size

        # if remainder_samples:
        #     batches += 1

        for idx in range(batches):
            images, labels, names = [], [], []
            # if idx == batches - 1:
            #     batch_idxs = indices[idx * batch_size:]
            # else:
            #     batch_idxs = indices[idx * batch_size:idx * batch_size +
            #                          batch_size]

            batch_idxs = indices[idx * batch_size:idx * batch_size +
                                 batch_size]

            batch_idxs = sorted(batch_idxs)

            imgs = itemgetter(*batch_idxs)(image_list)

            for img in imgs:

                # label = img.split('/')[-1].split('_')[0]
                label = img.split('/')[2]
                name = img.split('/')[-1]
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                flipper = iaa.Fliplr(1.0)
                image_flip = flipper.augment_image(img)

                img = np.array(img) / 255.0
                image_flip = np.array(image_flip) / 255.0

                images.append(img)
                images.append(image_flip)
                labels.append(to_categorical(label, 2))
                labels.append(to_categorical(label, 2))

            yield (np.array(images), np.array(labels))
            # images, labels, names = [], [], []


def model1():
    model = Sequential()
    model.add(
        Conv2D(
            32, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            input_shape=(100, 100, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(
        Conv2D(
            64, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(
        Conv2D(
            128, (3, 3),
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))

    # opt = Adam(lr=0.001)
    # model.compile(
    #     optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model2():
    inputs = Input(shape=(100, 100, 3))
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    # a = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    model.summary()

    return model


def xception():
    input_tensor = Input(shape=(100, 100, 3))
    model = keras.applications.xception.Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        pooling=None,
        classes=2)

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=x)
    # model.compile(
    #     optimizer='adam',
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])

    model.summary()

    return model


def resnet50():
    model = keras.applications.ResNet50(
        input_shape=(100, 100, 3),
        include_top=False,
        weights='imagenet',
        pooling=None,
        classes=2)

    model = build_finetune_model(model, [512, 1024], 2)

    # x = model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(2, activation='softmax')(x)
    # model = Model(inputs=model.input, outputs=x)
    # model.compile(
    #     optimizer='adam',
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # model.summary()

    return model


def efficientnet():
    model = efn.EfficientNetB5(
        include_top=False, weights='imagenet', input_shape=(100, 100, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)
    model.summary()

    return model


def build_finetune_model(model, fc_layers, num_classes):
    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    return model


def main():

    args = parser.parse_args()
    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        K.tensorflow_backend.set_session(tf.Session(config=config))

    image_list = glob.glob(os.path.join(IMG_PATH, '*.jpg'))

    sample_idxs = np.random.permutation(len(image_list))

    train_up_to = int(len(image_list) * TRAIN_VAL_SPLIT)
    train_sample_idx = sample_idxs[:train_up_to]
    validation_sample_idx = sample_idxs[train_up_to:]

    get_data_generator(image_list, train_sample_idx, batch_size=BATCH_SIZE)

    train_generator = get_data_generator(
        image_list, train_sample_idx, batch_size=BATCH_SIZE)
    validation_generator = get_data_generator(
        image_list, validation_sample_idx, batch_size=BATCH_SIZE)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            baseline=None,
            restore_best_weights=True),
        ModelCheckpoint(
            # f'models/{datetime.now().strftime("%m-%d-%Y-%H:%M")}-{val_loss:.2f}.h5',
            'models/{epoch:02d}-{val_loss:.2f}.h5',
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            mode="min")
    ]

    if args.model == 'xception':
        model = xception()
    elif args.model == 'VGG16_Inception':
        model = VGG16_Inception(include_top=False)
    elif args.model == 'Resnet50':
        model = resnet50()
    elif args.model == 'efficientnet':
        model = efficientnet()
    elif args.model == 'model2':
        model = model2()
    else:
        model = model1()

    adam = Adam(lr=0.001)
    model.compile(
        optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # print(f'model layers:{len(model.layers)}')

    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=len(train_sample_idx) // BATCH_SIZE,
        validation_steps=len(validation_sample_idx) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks)


if __name__ == "__main__":
    main()