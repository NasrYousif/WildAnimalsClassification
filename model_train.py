from keras import losses, metrics
from build_model import build_model


def train_model(train_data, validation):
    """
    The function is used to train the model using labeled data
    :param train_data: String
           Dataset format used to train the model
    :param validation: String
           Dataset format used to evaluate the model
    :return:
    """

    # load the neural network
    model = build_model()

    # compile the model
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(),
                  metrics=metrics.BinaryAccuracy(
                      name="binary_accuracy", dtype=None, threshold=0.5
                  ))
    # train the model
    model.fit(train_data, batch_size=64,
              validation_data=validation,
              epochs=10)

    # save the trained model into outputs folder
    model.save('./outputs/savedModel')
