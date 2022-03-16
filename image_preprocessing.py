from keras.preprocessing.image_dataset import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator


def image_processing(train_directory: object, test_directory: object):
    """
    This function load the image from directory
       :param train_directory: string
            Path into training data
       :param test_directory: string
            Path into test data
       :return:
       data_generator : dataset format loaded
    """
    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.25,
        horizontal_flip=True,
    )

    train_generator = datagen.flow_from_directory(
        train_directory,
        class_mode="binary",
        color_mode="rgb",
        subset='training',
        target_size=(128, 128),
        shuffle=True,
        batch_size=64,
        seed=20)

    valid_generator = datagen.flow_from_directory(
        train_directory,
        class_mode="binary",
        color_mode="rgb",
        batch_size=64,
        subset='validation',
        target_size=(128, 128),
        shuffle=True,
        seed=20)

    test_generator = image_dataset_from_directory(
        test_directory,
        color_mode="rgb",
        shuffle=False,
        label_mode=None,
        image_size=(128, 128),
        batch_size=32,
    )

    return train_generator, valid_generator, test_generator
