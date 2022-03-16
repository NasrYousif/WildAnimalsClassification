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

image = Image.open('path/to/image')</code>
### 3. Check your realtime prediction
Run <code>recognizer.py</code> and look at the camera to see if your face recognized well, 

## References
- [OpenCV Library](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0)
