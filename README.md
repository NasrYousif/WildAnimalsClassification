# Wild Animals Classification
This model is using dataset from zindi.africa platform to classifiy two kind to famous animal images in africa epephants and zebra, so I have build a computer vision project that can identify with image for zrbra or elephant.  

## Dataset citation:
The data used to train the model have been downloaded from this online competetion platfrom : https://zindi.africa/competitions/sbtic-animal-classification  

### classes
<code>class 0 : Zebra</code>   <br/>
<code>class 1 : Elephant</code>

![Class A](https://github.com/NasrYousif/ZebraElephantClassification/blob/master/assets/zepra.jpeg) ![Class A](https://github.com/NasrYousif/ZebraElephantClassification/blob/master/assets/Elephant.jpeg)  

## Steps to make prediction
### 1. install dependancies
<code>pip install Pillow numpy</code>
### 2. Download Image.
you can download any image from online sources
### 3. convert the image into array
<code>from PIL import Image
import numpy as np
image = Image.open('path/to/image')</code>

numpy_img = np.array(image)
### 4. Make prediction
use code below to load in predict your target  
<code>
    # load the model
    model = keras.models.load_model('./outputs/savedModel')
    # predict the targets using predict function un keras
    y_pred = (model.predict(test_data) > 0.5).astype("int32")
</code>  

## References
- [TensorFlow](tensorflow.org)
- [SciPy ](https://scipy.org/)
