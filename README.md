# Wild Animals Classification
This model is using dataset from zindi.africa platform to classifiy two kind to famous animal images in africa epephants and zebra, so I have build a computer vision project that can identify with image for zrbra or elephant.  

## Dataset citation:
The data used to train the model have been downloaded from this online competetion platfrom : https://zindi.africa/competitions/sbtic-animal-classification
<br/>
<code>class 0 : Zebra</code>   <br/>
<code>class 1 : Elephant</code>

![Class A](https://github.com/NasrYousif/ZebraElephantClassification/blob/master/assets/zepra.jpeg) ![Class A](https://github.com/NasrYousif/ZebraElephantClassification/blob/master/assets/Elephant.jpeg)  

## Steps to train your custom system
### 1. Make a custom dataset.
Run <code>make_dataset.py</code> to open your camera and make shots by preesing the key <code> "k" </code> for every shot for every user face that you want to recognize, after finish press <code> "q" </code> to close the window.\n
### 2. Train your model
Open <code>train_model.py</code> and run it to generate your <code>trainer.yml</code> file contain 128-D array to reconize your faces.
### 3. Check your realtime prediction
Run <code>recognizer.py</code> and look at the camera to see if your face recognized well, 

## References
- [OpenCV Library](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0)
