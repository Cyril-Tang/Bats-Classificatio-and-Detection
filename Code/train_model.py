import tensorflow as tf
import pathlib
import random
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator



data_root = pathlib.Path.home().joinpath('Dissertation/Data/')
for item in data_root.iterdir():
   print(item)

all_image_paths = list(data_root.glob('*/*.jpg'))
all_image_paths = [str(path) for path in all_image_paths if "DS_Store" not in str(path)]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = []
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                   for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

img_size = (224, 224)
def load_and_preprocess_image(image):
   image = cv2.imread(image)
   image = cv2.resize(image, img_size)
   image = image[:, :, [2, 1, 0]]
   image = image.astype('float64')
   image /= 255.0  # normalize to [0,1] range

   return image

example_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(example_path))
plt.title(label_names[label].title())
print()

images = []
for image_path in tqdm(all_image_paths):
   processed_image = load_and_preprocess_image(image_path)
   images.append(processed_image)
images = np.asarray(images)

all_image_labels = np.asarray(all_image_labels)
all_image_labels = to_categorical(all_image_labels, num_classes=13)

np.save('images.npy', images)
np.save('all_image_labels.npy', all_image_labels)

images = np.load('images.npy')
all_image_labels = np.load('all_image_labels.npy')

X_train, X_val, y_train, y_val = train_test_split(images, all_image_labels, test_size=0.2, random_state=3)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

callbacks = [ 
tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_loss'),
tf.keras.callbacks.TensorBoard(log_dir='./logs'),
tf.keras.callbacks.LearningRateScheduler(scheduler)]

base_model0 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model1 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model2 = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model4 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_models = [base_model0, base_model1, base_model2, base_model3, base_model4]

trained_historise = []
trained_models = []

FC_NUMS = 1024
IMAGE_SIZE = 224
NUM_CLASSES = 13
TRAINABLE_LAYERS = 2

for i, base_model in enumerate(base_models):
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_NUMS, activation='relu')(x)
    x = Dropout(0.05)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.05)(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.05)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.05)(x)
    prediction = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    model.summary()
    print("layer nums:", len(model.layers))

    for layer in model.layers[:TRAINABLE_LAYERS]:
        layer.trainable = True
    for layer in model.layers:
        print("layer.trainable:", layer.trainable)

    LEARNING_RATE = 1e-3
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 100
    batch_size = 16

    history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=2,
                    validation_data = (X_val, y_val),
                   callbacks=callbacks)
    
    trained_historise.append(history)
    trained_models.append(model)
    model.save('model/model_num'+str(i)+'.h5')



for model in model_history:
    print("**********************************")
    result = model.evaluate(X_val, y_val, verbose=0)
    print("loss: {}".format(result[0]))
    print("accuracy: {}".format(result[1]))


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,31))
ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")