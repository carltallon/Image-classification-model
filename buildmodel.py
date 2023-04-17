#dependencies
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

#ensure GPU's only use required memory
print("You should uncomment GPU alteration \n")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#creating dataset and image resizing 256x256, shuffle, batching them into batches of 32
#building data pipleine
print("Creating dataset and image resizing 256x256, shuffle, batching them into batches of 32..\n")
data = tf.keras.utils.image_dataset_from_directory('shipimages')

#convert data into numpy iterator 
#allowing us to access/loop through data pipeline
print("Convert data into numpy iterator..\n")
data_iterator = data.as_numpy_iterator()

#creating batch of data 
print("Creating batch of data.. \n")
batch = data_iterator.next()

fig,ax = plt.subplots(ncols=4, figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.savefig('shipclassification.png')


#scale data within pipeline
print("Scaling data within pipeline.. \n")
data = data.map(lambda x,y:(x/255,y))

# find min and max values (test to see if images are valid)
scaled_iterator = data.as_numpy_iterator()
print("Min value = " ,scaled_iterator.next()[0].min())
print("Max value = " ,scaled_iterator.next()[0].max())


#finding length of data
print("Length of data = " , len(data))

#set train valuation and test variables
# used to train deep learning models
train_size = int(len(data)*.7)
print("Train = " , train_size)
#used to evaluate model during training
val_size = int(len(data)*.2)
print("Valuation = " ,val_size)
#Used in final evalutaion stage
test_size = int(len(data)*.1)
print("Test = ",test_size)

# partitiong data and allocating batches to data
# CREATING PARTITIONS FOR MODELING
print("Allocating batches to data.. \n ")
train = data.take(train_size)
#skip batches that have already been allocated to train 
val = data.skip(train_size).take(val_size)
#skip batches that have already been allocated to train and val 
test = data.skip(train_size+val_size).take(test_size)

#import DLM dependencies
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#create sequential model instance
model = Sequential()

# add layers SEQUENTIALLY 
print("Adding layers.. \n")
# add convolution layer and max pooling layer
#16 FILTERS, 3X3 PIXELS, STRIDE OF 1 (MOVE 1 PIXEL AT TIME), RELU ACTIVATION = TAKING OUTPUT AND PASSING THROUGH RELO FUNCTION (_/), DATA SHAPE
model.add(Conv2D(16, (3,3), 1, activation= 'relu', input_shape=(256,256,3)))
#take maximum value after relu activation and return value
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation= 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation= 'relu'))
model.add(MaxPooling2D())

#flatten model
model.add(Flatten())

#fully connected layers (dense layers)
model.add(Dense(256, activation = 'relu'))

#ensuring single output ( 0 or 1 )
model.add(Dense(1, activation = 'sigmoid'))

#COMPILE MODEL
#USING ADAM OPTIMISER, loss in binary format and tracking accuracy metric
print("Compiling model.. \n")
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
#show model summary (Neural network)
model.summary()

#create log directory variables
logdir = 'logs'
#log out as model trains
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# TRAIN MODEL
# EPOCHS = HOW LONG TO TRAIN FOR, 1 EPOCH = 1 PASS OVER TRAINING DATA
print("Training model... \n")
hist = model.fit(train, epochs=2, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
plt.legend(loc='upper left')
plt.savefig('shipgraph.png')


from tensorflow.python.keras.metrics import Precision, Recall, BinaryAccuracy

# MODEL Evaluation and testing
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x , y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print("Pre = " ,pre.result())
print("Re = " ,re.result())
print("Acc = " ,acc.result())

#SAVE MODEL
print("Saving model.. \n")
model.save(os.path.join('models', 'shipmodel.h5'))