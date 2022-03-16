from tensorflow import keras


def predict(test_data):
    """
    This function is make predictions for the system from the saved model in output directory
    :param test_data: tensorflow dataset format
          Test data the need to predict its targets
    :return: array
           The predicted Target for the unlabeled data
    """

    # load the model
    model = keras.models.load_model('./outputs/savedModel')

    # predict the targets using predict function un keras
    y_pred = (model.predict(test_data) > 0.5).astype("int32")

    return y_pred
