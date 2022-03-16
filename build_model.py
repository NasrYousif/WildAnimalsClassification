from keras import models, layers


def build_model() -> object:
    """
    This function build a neural network
    :return: model
    """
    # feature extractor
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))

    # model trainer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    return model
