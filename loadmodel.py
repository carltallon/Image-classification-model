#dependencies
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

#Load model from memory
print("Loading model from memory.. \n")
new_model = load_model(os.path.join('models', 'shipmodel.h5'))

#Model Testing
#import image to test
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
img = cv2.imread('noship.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


#pass image to model
print("Passing image to model.. \n")
yhat = new_model.predict(np.expand_dims(resize/255,0))
print(yhat)

if yhat > 0.5:
    print('Image does contain a ship')
else:
    print('Image does not contain a ship')