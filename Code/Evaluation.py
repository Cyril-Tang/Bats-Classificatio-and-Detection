import tensorflow as tf
import pathlib
import random
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json as js
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

bat_imgs = np.load('bat_imgs.npy')
bat_labels = np.load('bat_labels.npy')

NUM_CLASSES = 6

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

callbacks = [
# tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
tf.keras.callbacks.TensorBoard(log_dir='./logs'),
tf.keras.callbacks.LearningRateScheduler(scheduler)]
epochs = 40
batch_size = 16

# split data into training data and testing data
X_train, X_val, y_train, y_val = train_test_split(bat_imgs, bat_labels, test_size=0.1, random_state=3)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

#=========================== hidden units =================================
trained_histories = []
units_list = [16, 32, 64, 128, 256]

for dense_num in units_list:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.05)(x)
    x = Dense(dense_num, activation='relu')(x)
    x = Dropout(0.05)(x)
    prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    LEARNING_RATE = 1e-3
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=2,
                    validation_data = (X_val, y_val),
                   callbacks=callbacks)
    trained_histories.append(history)
    with open('Effect/units'+str(dense_num)+'.txt', 'a') as f:
        f.write(js.dumps(str(history.history)))
        f.close()

epoch_list = list(range(1,epochs+1))
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("training accuracy")

for i, history in enumerate(trained_histories):
    print(i)
    plt.plot(epoch_list, history.history['accuracy'], label="hidden units "+str(units_list[i]), linestyle='--')
plt.legend(loc='lower right')

plt.savefig('Effect/training_accuracy_hidden_unit.png')

plt.subplot()
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
for i, history in enumerate(trained_histories):
    plt.plot(epoch_list, history.history['val_accuracy'], label="hidden units "+str(units_list[i]), linestyle='--')
plt.legend(loc='lower right')
plt.savefig('Effect/validation_accuracy_hidden_unit.png')


#============================ batch size ==========================
trained_histories = []
batch_list = [4, 8, 16, 32, 64]

for batch_num in batch_list:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.05)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.05)(x)
    prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    LEARNING_RATE = 1e-3
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, batch_size=batch_num,
                    epochs=epochs, verbose=2,
                    validation_data = (X_val, y_val),
                   callbacks=callbacks)
    trained_histories.append(history)

    with open('Effect/batch'+str(batch_num)+'.txt', 'a') as f:
        f.write(js.dumps(str(history.history)))
        f.close()

epoch_list = list(range(1,epochs+1))
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("training accuracy")

for i, history in enumerate(trained_histories):
    plt.plot(epoch_list, history.history['accuracy'], label="batch size "+str(batch_list[i]), linestyle='--')
plt.legend(loc='lower right')

plt.savefig('Effect/training_accuracy_batch_num.png')

plt.subplot()
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
for i, history in enumerate(trained_histories):
    plt.plot(epoch_list, history.history['val_accuracy'], label="batch size "+str(batch_list[i]), linestyle='--')
plt.legend(loc='lower right')
plt.savefig('Effect/validation_accuracy_batch_num.png')

#=========================== Optimizer ==================================
trained_histories = []
opt_list = [Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False), SGD(learning_rate=LEARNING_RATE, momentum=0.0, nesterov=False), RMSprop(learning_rate=LEARNING_RATE, rho=0.9)]
i = 0
for opt in opt_list:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.05)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.05)(x)
    prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    LEARNING_RATE = 1e-3
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, batch_size=16,
                    epochs=epochs, verbose=2,
                    validation_data = (X_val, y_val),
                   callbacks=callbacks)
    trained_histories.append(history)

    with open('Effect/OPT'+str(i)+'.txt', 'a') as f:
        f.write(js.dumps(str(history.history)))
        f.close()
    i += 1

epoch_list = list(range(1,epochs+1))
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("training accuracy")

for i, history in enumerate(trained_histories):
    plt.plot(epoch_list, history.history['accuracy'], label="Optimizer "+str(i), linestyle='--')
plt.legend(loc='lower right')

plt.savefig('Effect/training_accuracy_optimizer.png')

plt.subplot()
plt.figure(figsize=(10,5), dpi=100)
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
for i, history in enumerate(trained_histories):
    plt.plot(epoch_list, history.history['val_accuracy'], label="Optimizer "+str(i), linestyle='--')
plt.legend(loc='lower right')
plt.savefig('Effect/validation_accuracy_optimizer.png')
