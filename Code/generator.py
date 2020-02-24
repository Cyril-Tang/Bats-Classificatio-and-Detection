import tensorflow as tf
import os
# 选择编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pathlib

data_root = pathlib.Path('Data/train_img')
for item in data_root.iterdir():
    print(item)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels =[]
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:5])

import numpy as np
# np.save('all_image_paths.npy', all_image_paths)
all_image_paths = np.asarray(all_image_paths)

# One hot vector representation of labels
all_image_labels = np.asarray(all_image_labels)
from keras.utils import to_categorical
all_image_labels = to_categorical(all_image_labels, num_classes=2)

# saving the y_labels_one_hot array as a .npy file
# np.save('all_image_labels.npy', all_image_labels)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(all_image_paths, all_image_labels, test_size=0.2, random_state=6)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

# base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
# base_model = VGG19(include_top=False, input_shape=(224, 224, 3))
# base_model = MobileNet(include_top=False, input_shape=(224, 224, 3))
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3))
# base_model = InceptionV3(include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

FC_NUMS = 512
IMAGE_SIZE = 224
# FREEZE_LAYERS = 17
NUM_CLASSES = 2

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_NUMS, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

from tensorflow.keras.models import Model

# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)
# 可观察模型结构
model.summary()
# 获取模型的层数
print("layer nums:", len(model.layers))
total_layers = len(model.layers)


for layer in model.layers[:total_layers//16]:
    layer.trainable = True
for layer in model.layers[total_layers-total_layers//16:]:
    layer.trainable = True
for layer in model.layers:
    print("layer.trainable:", layer.trainable)

from tensorflow.keras.optimizers import SGD, Adam

LEARNING_RATE = 1e-3
model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


callbacks = [
tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
tf.keras.callbacks.TensorBoard(log_dir='./logs'),
tf.keras.callbacks.LearningRateScheduler(scheduler)]

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

train_dir = 'Data/train_img/'
 
num_epochs = 100
batch_size = 64
 
data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)
 
# classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None.
# 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。
# 每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。
# 通过属性class_indices可获得文件夹名与类的序号的对应字典。
# 本例中使用默认的参数，表示按数字或字母升序，对应类的序号
train_generator = data_gen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=batch_size,
                                               class_mode='categorical', subset='training')
validation_generator = data_gen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=batch_size,
                                               class_mode='categorical', subset='validation')

 
# 训练模型
history = model.fit_generator(generator=train_generator,
                    epochs=num_epochs,
                    verbose=2,
                    validation_data=validation_generator,
                    callbacks=callbacks)

model.save('ResNet50.h5') 

from matplotlib import pyplot as plt
plt.title('Keras model loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# Visualize History for Accuracy.
plt.title('Keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc='lower right')
plt.show()