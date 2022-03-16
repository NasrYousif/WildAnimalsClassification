from image_preprocessing import image_processing
from model_train import train_model
from model_predict import predict
from pandas import read_csv

# refer to the images directory
train_directory = 'images/train/'
test_directory = 'images/test'

# prepare the images
train, valid, test = image_processing(train_directory, test_directory)

# train the model
train_model(train, valid)

# make prediction
y_pred = predict(test)
print(y_pred)
# load the submission file
sub = read_csv('sample_submission.csv')

# map the prediction into label column
sub['label'] = y_pred
# show before saved
sub.head()
# save the predictions in final file
sub.to_csv('final_submission.csv', index=False)
