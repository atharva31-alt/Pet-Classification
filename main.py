import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
import glob
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ppi
from tensorflow.keras.layers import RandomFlip, RandomRotation, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

images_fp = "./images"
images_names = [os.path.basename(file) for file in glob.glob(os.path.join(images_fp, '*.jpg'))]

labels = [' '.join(name.split('_')[:-1:]) for name in images_names]

def label_encode(label):
    if label == 'abyssinian': return 0
    elif label == 'american shorthair': return 1
    elif label == 'beagle': return 2
    elif label == 'boxer': return 3
    elif label == 'bulldog': return 4
    elif label == 'chihuahua': return 5
    elif label == 'corgi': return 6
    elif label == 'dachshund': return 7
    elif label == 'german shephard': return 8
    elif label == 'golden retriever': return 9
    elif label == 'husky': return 10
    elif label == 'labrador': return 11
    elif label == 'maine coon': return 12
    elif label == 'mumbai cat': return 13
    elif label == 'persian cat': return 14
    elif label == 'pomeranian': return 15
    elif label == 'pug': return 16
    elif label == 'ragdoll cat': return 17
    elif label == 'rottwiler': return 18
    elif label == 'shiba inu': return 19
    elif label == 'siamese cat': return 20
    elif label == 'sphynx': return 21
    elif label == 'yorkshire terrier': return 22

features = []
labels = []
IMAGE_SIZE = (224, 224)

for name in images_names:
    label = ' '.join(name.split('_')[:-1:])
    label_encoded = label_encode(label)
    if label_encoded != None:
        img = load_img(os.path.join(images_fp, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        img_array = np.array(img)
        features.append(img_array)
        labels.append(label_encoded)

features_array = np.array(features)
labels_array = np.array(labels)

labels_one_hot = pd.get_dummies(labels_array)

# train
MODEL_PATH = "pet_model.keras"

x_train, x_test, y_train, y_test = train_test_split(features_array, labels_one_hot, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

model_history = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    num_classes = y_train.shape[1]
    data_augmentation = Sequential([RandomFlip("horizontal_and_vertical"), RandomRotation(0.2)])

    features_array = ppi(features_array.astype('float32'))

    resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    resnet_model.trainable = False

    # building model
    inputs = Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = resnet_model(x, training=False)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    model_history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=10)

    model.save("pet_model.keras")

if model_history is not None:
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_range = range(10)
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training accuracy')
    plt.plot(epochs_range, val_acc, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
print(y_pred)