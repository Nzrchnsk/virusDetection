import os
import struct

import numpy as np

from PIL import Image
from ClassificationModel.dataset_utils import prep_dataset

import cv2

from ClassificationModel.dataset_utils import prep_file


def file_to_bytes_old(file_path: str) -> [bytes]:
    byte_arr = []
    with open(file_path, "rb") as binary_file:
        while True:
            row = binary_file.read(128)

            if not row:
                break

            unpack_byte = list(struct.unpack(f'>{len(row)}B', row))
            byte_arr.append(unpack_byte)
    if len(byte_arr[-1]) < 128:
        zeros_vec = list(np.zeros(128 - len(byte_arr[-1])))
        byte_arr[-1].extend(zeros_vec)
    return byte_arr


def file_to_bytes(file_path: str) -> [bytes]:
    byte_arr = []
    rep = 0x00000000
    i = 0
    with open(file_path, "rb") as binary_file:
        while True:
            row = binary_file.read(1)

            if not row:
                break

            rep = rep | row[0]
            i += 1
            if i >= 1:
                byte_arr.append(rep)
                rep = 0x00000000
                i = 0
            else:
                rep = rep << 1

    return byte_arr


def rename_all_files_in_directory(directory_path: str, name_pattern: str = None):
    files = os.listdir(directory_path)
    i = 0
    for file in files:
        os.rename(f'{directory_path}/{file}', f'{directory_path}/{name_pattern}{i}')
        i += 1


def prepare_dataset_file_names():
    rename_all_files_in_directory('train/train/mal', 'mal')
    rename_all_files_in_directory('train/train/ben', 'ben')


# def convert_file_to_img_old(file_data: []) -> Image:
#     file_data = np.array(file_data, dtype=np.int32)
#     img = file_data
#     img.resize((10157, 128), refcheck=False)
#     img = Image.fromarray(img).convert('L')
#     return img

def convert_file_to_img(file_data: np.ndarray, shape: tuple) -> Image:
    img = file_data
    img = img.reshape(shape)
    img = Image.fromarray(img)
    return img


def convert_all_dataset_exe_to_img_old():
    mal_simples_dir = 'train/train/mal'
    ben_simples_dir = 'train/train/ben'
    img_mal_simples_dir = 'train/images/mal'
    img_ben_simples_dir = 'train/images/ben'

    mal_simples = os.listdir(mal_simples_dir)
    ben_simples = os.listdir(ben_simples_dir)
    for simple in mal_simples:
        img = convert_file_to_img(file_to_bytes(f'{mal_simples_dir}/{simple}'))
        img.save(f'{img_mal_simples_dir}/{simple}.png')
        # try:
        #     img = convert_exe_to_cv2_img(prep_file.file_to_bytes(f'{mal_simples_dir}/{simple}'))
        #     cv2.imwrite(f'{img_mal_simples_dir}/{simple}.png', img)
        # except:
        #     print(f'{mal_simples_dir}/{simple}')
        #     continue
    print('convertion malvare to images succeded')
    for simple in ben_simples:
        img = convert_file_to_img(file_to_bytes(f'{ben_simples_dir}/{simple}'))
        img.save(f'{img_ben_simples_dir}/{simple}.png')
        # try:
        #     img = convert_exe_to_cv2_img(prep_file.file_to_bytes(f'{ben_simples_dir}/{simple}'))
        #     cv2.imwrite(f'{img_ben_simples_dir}/{simple}.png', img)
        # except:
        #     print(f'{ben_simples_dir}/{simple}')
        #     continue
    print('convertion benings to images succeded')


def convert_all_dataset_exe_to_img():
    mal_simples_dir = 'train/train/mal'
    ben_simples_dir = 'train/train/ben'
    img_mal_simples_dir = 'train/images/mal'
    img_ben_simples_dir = 'train/images/ben'

    mal_simples = os.listdir(mal_simples_dir)
    ben_simples = os.listdir(ben_simples_dir)
    for simple in mal_simples:
        byte_array = file_to_bytes(f'{mal_simples_dir}/{simple}')
        byte_array_np = np.asarray(byte_array, np.uint8)
        # w = 120
        # h = int(len(byte_array) / w)
        # if len(byte_array) - h * w != 0:
        #     h += 1

        # if h * w * 3 != len(byte_array_np):
        #     byte_array_np.resize((1, h * w * 3), refcheck=False)
        byte_array_np.resize((1, 1503792), refcheck=False)
        # img = convert_file_to_img(file_data=byte_array_np, shape=(h, w, 3))
        img = convert_file_to_img(file_data=byte_array_np, shape=(708, 708, 3))
        img.save(f'{img_mal_simples_dir}/{simple}.png')
        # img.save('tmp.png')
        # try:
        #     img = convert_exe_to_cv2_img(prep_file.file_to_bytes(f'{mal_simples_dir}/{simple}'))
        #     cv2.imwrite(f'{img_mal_simples_dir}/{simple}.png', img)
        # except:
        #     print(f'{mal_simples_dir}/{simple}')
        #     continue
    print('convertion malvare to images succeded')
    for simple in ben_simples:
        byte_array = file_to_bytes(f'{ben_simples_dir}/{simple}')
        byte_array_np = np.asarray(byte_array, np.uint8)
        byte_array_np.resize((1, 1503792), refcheck=False)
        img = convert_file_to_img(file_data=byte_array_np, shape=(708, 708, 3))
        img.save(f'{img_ben_simples_dir}/{simple}.png')

    print('convertion benings to images succeded')


def build_model_grayscale():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
    from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
    from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), name='conv0', input_shape=(10157, 128, 1)))

    model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(300, activation="relu", name='rl'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', name='sm'))
    print(model.summary())
    return model


def build_model():
    from tensorflow.keras import layers
    from tensorflow.keras import Model

    # Our input feature map is 128x150: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(708, 708, 3))

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Create output layer with a single node and sigmoid activation
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = Model(img_input, output)
    print(model.summary())

    return model


def prepare_dataset():
    prep_dataset.main()


def tensor_flow_settings():
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    from tensorflow.python.framework.config import set_memory_growth
    tf.compat.v1.disable_v2_behavior()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def data_preprocessing(train_dir: str, validation_dir: str):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255
    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_data_gen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(708, 708),  # All images will be resized to 150x150
        batch_size=1,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow validation images in batches of 20 using val_datagen generator
    validation_generator = val_data_gen.flow_from_directory(
        validation_dir,
        target_size=(708, 708),
        batch_size=1,
        class_mode='binary')

    return train_generator, validation_generator


def test():
    import cv2
    from ClassificationModel.dataset_utils import prep_file
    file_data = prep_file.file_to_bytes(file_path='train/train/ben/ben10')
    file_data_np = np.fromstring(str(file_data), np.uint8)
    img_np = cv2.imdecode(file_data_np, flags=1)
    print(img_np)
    print(type(img_np))
    # cv2.imwrite('0.png', img_np)


def convert_exe_to_cv2_img(file_data: []):
    file_data = np.array(file_data, dtype=np.uint8)
    img = file_data
    img.resize((1, file_data.size), refcheck=False)

    img = Image.fromarray(img).convert('L')
    img.save('tmp.png')
    img = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (1, 1000), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('0cv2.png', img)
    return img


def main():
    from tensorflow import device
    with device('/gpu:0'):
        tensor_flow_settings()
        model = build_model()
        # model = build_model_grayscale()
        from tensorflow.keras.optimizers import RMSprop

        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['acc'])
        train_generator, validation_generator = data_preprocessing(
            train_dir='dataset/train',
            validation_dir='dataset/validation')

        history = model.fit(
            train_generator,
            steps_per_epoch=321,  # 2000 images = batch_size * steps
            epochs=15,
            validation_data=validation_generator,
            validation_steps=53,  # 1000 images = batch_size * steps
            verbose=1)


def test_cv_resize():
    # img = Image.open('dataset_big_size/train/mal/mal6.png')
    # img = img.copy()
    # img = img.resize((128, 128))
    # img.save('tmp.png')
    img = cv2.imread('dataset_big_size/train/mal/mal6.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1, 10000), interpolation=cv2.INTER_AREA)
    cv2.imwrite('1.png', img)


if __name__ == "__main__":
    main()
