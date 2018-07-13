import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import urllib3

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

image_name = "tree3"

#<---------------------------Download-Image--------------------------->               
http = urllib3.PoolManager()
r = http.request('GET', 'https://thumbs.dreamstime.com/b/green-tree-white-background-view-31183303.jpg', preload_content=False)

with open(f'TFTut/Downloaded_images/{image_name}.jpg', 'wb') as out:
    while True:
        data = r.read(1024)
        if not data:
            break
        out.write(data)

r.release_conn()
#<---------------------------Import-Image---------------------------->
img = cv2.imread(f'/home/neo/Code/TFTut/Downloaded_images/{image_name}.jpg') 
#<---------------------------Image-Preprocessing---------------------------->
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
width = 28
height = 28

dim = (width, height)
test_image = cv2.resize((255-gray_img), dim, interpolation = cv2.INTER_AREA)

cv2.imwrite(f'{image_name}_processed.jpg',test_image)

image_set = (np.expand_dims(test_image,0))
test_images = image_set/ 255.0

#<---------------------------Import-Model---------------------------->

new_model = keras.models.load_model('/home/neo/Code/TFTut/Saved_Model/saved_model_200.h5')

#<---------------------------Predictions---------------------------->

predictions = new_model.predict(test_images)
predicted_label = class_names[np.argmax(predictions)]
predicted_probability = np.max(predictions)
print(f"{predicted_label}, Probability : {predicted_probability}")




