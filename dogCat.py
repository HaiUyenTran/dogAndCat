import os, shutil
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

def extract_feature(dir, sample_count):
    feature  = np.zeros(shape=(sample_count, 4, 4, 512))
    label = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(dir, target_size=(150,150),
                                            batchSize = batch_size,
                                            class_mode = 'binary')
    i = 0
    for inputs_batch, label_batch in generator:
        feature_batch = convBase.predict(inputs_batch)
        feature[i * batch_size : (i + 1) * batch_size] = feature_batch
        label[i * batch_size : (i + 1) * batch_size] = label_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return feature,label


def preprocessing_data(srcDir,orDir):
    # Create folder
    baseDir = orDir+"/cats_and_dogs_small"
    os.mkdir(baseDir)

    trainDir = os.path.join(baseDir,"train")
    os.mkdir(trainDir)

    valDir = os.path.join(baseDir,"validation")
    os.mkdir(valDir)

    testDir = os.path.join(baseDir,"test")
    os.mkdir(testDir)

    trainCatDir = os.path.join(trainDir,"cats")
    os.mkdir(trainCatDir)

    trainDogDir = os.path.join(trainDir,"dogs")
    os.mkdir(trainDogDir)

    valCatDir = os.path.join(valDir,"cats")
    os.mkdir(valCatDir)

    valDogDir = os.path.join(valDir,"dogs")
    os.mkdir(valDogDir)

    testCatDir = os.path.join(testDir,"cats")
    os.mkdir(testCatDir)

    testDogDir = os.path.join(testDir,"dogs")
    os.mkdir(testDogDir)
    # Copy image to folder
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(trainCatDir, fname)
        shutil.copyfile(src,dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(trainDogDir, fname)
        shutil.copyfile(src,dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(valCatDir, fname)
        shutil.copyfile(src,dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(valDogDir, fname)
        shutil.copyfile(src,dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(testCatDir, fname)
        shutil.copyfile(src,dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(srcDir, fname)
        dst = os.path.join(testDogDir, fname)
        shutil.copyfile(src,dst)

# preprocessing_data("E://DL//dogandcat_deeplearningwithpython//dogs-vs-cats//train//train", "E://DL//dogCatPractice")
baseDir = "cats_and_dogs_small"
trainDir = os.path.join(baseDir,"train")
testDir = os.path.join(baseDir,"test")
valDir = os.path.join(baseDir,"validation")

trainCatDir = os.path.join(trainDir,"cats")
testCatDir = os.path.join(testDir,"cats")
valCatDir = os.path.join(valDir,"cats")

trainDogDir = os.path.join(trainDir,"dogs")
testDogDir = os.path.join(testDir,"dogs")
valDogDir = os.path.join(valDir,"dogs")

#create convBase
convBase = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
convBase.summary()

#extract feature
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10

train_features, train_labels = extract_feature(trainDir,2000)
val_features, val_labels = extract_feature(testDir,1000)
test_features, test_labels = extract_feature(testDir,1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(val_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# Defining and training the densely connected classifier
model = models.Sequential()
model.add(convBase)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Data gen
train_datagen = ImageDataGenerator( rescale= 1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode= 'nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(trainDir,
                                                    target_size=(150,150),
                                                    batch_size = 20,
                                                    class_mode='binary')
val_generator = test_datagen.flow_from_directory(valDir,
                                                 target_size=(150,150),
                                                 batch_size = 20,
                                                 class_mode="binary")



model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch= 100,
                              epochs = 5,
                              validation_data=val_generator,
                              validation_steps= 50)
import keras as kr
kr.save_model_hdf5(model, "cats_and_dogs_small_3.h5")
#plot the result
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print("done")
