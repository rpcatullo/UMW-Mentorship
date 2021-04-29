import os
import gc
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

# 1.) Get all image path sets
directory = 'C:/Users/rcatu/Downloads/Spectrograms-20210429T073650Z-001/Spectrograms/{}'
directory0 = 'C:/Users/rcatu/Downloads/Spectrograms-20210429T073650Z-001/Spectrograms'
alfaroi_set = [directory.format(i) for i in os.listdir(directory0) if 'alfaroi' in i]
cinerascens_set = [directory.format(i) for i in os.listdir(directory0) if 'cinerascens' in i]
lanciformis_set = [directory.format(i) for i in os.listdir(directory0) if 'lanciformis' in i]
bifurcus_set = [directory.format(i) for i in os.listdir(directory0) if 'bifurcus' in i]
petersi_set = [directory.format(i) for i in os.listdir(directory0) if 'petersi' in i]
discodactylus_set = [directory.format(i) for i in os.listdir(directory0) if 'discodactylus' in i]
fuscifacies_set = [directory.format(i) for i in os.listdir(directory0) if 'fuscifacies' in i]
conspicillatus_set = [directory.format(i) for i in os.listdir(directory0) if 'conspicillatus' in i]
margaritifer_set = [directory.format(i) for i in os.listdir(directory0) if 'margaritifer' in i]

# 2.) Randomly Shuffle Images Before Splitting for Training and Testing
random.shuffle(alfaroi_set)
random.shuffle(cinerascens_set)
random.shuffle(lanciformis_set)
random.shuffle(bifurcus_set)
random.shuffle(petersi_set)
random.shuffle(discodactylus_set)
random.shuffle(fuscifacies_set)
random.shuffle(conspicillatus_set)
random.shuffle(margaritifer_set)

# 3.) Training and Testing Image Sets
alfaroi_train, alfaroi_test = train_test_split(alfaroi_set, test_size=0.25, random_state=42)
cinerascens_train, cinerascens_test = train_test_split(cinerascens_set, test_size=0.25, random_state=42)
lanciformis_train, lanciformis_test = train_test_split(lanciformis_set, test_size=0.25, random_state=42)
bifurcus_train, bifurcus_test = train_test_split(bifurcus_set, test_size=0.25, random_state=42)
petersi_train, petersi_test = train_test_split(petersi_set, test_size=0.25, random_state=42)
discodactylus_train, discodactylus_test = train_test_split(discodactylus_set, test_size=0.25, random_state=42)
fuscifacies_train, fuscifacies_test = train_test_split(fuscifacies_set, test_size=0.25, random_state=42)
conspicillatus_train, conspicillatus_test = train_test_split(conspicillatus_set, test_size=0.25, random_state=42)
margaritifer_train, margaritifer_test = train_test_split(margaritifer_set, test_size=0.25, random_state=42)

train_set = alfaroi_train + cinerascens_train + lanciformis_train + bifurcus_train + petersi_train + discodactylus_train + fuscifacies_train + conspicillatus_train + margaritifer_train
test_set = alfaroi_test + cinerascens_test + lanciformis_test + bifurcus_test + petersi_test + discodactylus_test + fuscifacies_test + conspicillatus_test + margaritifer_test
#train_set, test_set = train_test_split(alfaroi_set + cinerascens_set + lanciformis_set + bifurcus_set + petersi_set + discodactylus_set + fuscifacies_set + conspicillatus_set + margaritifer_set, test_size = 0.25, random_state=42)

# 4.) Garbage Collection
del alfaroi_set, cinerascens_set, lanciformis_set, bifurcus_set, petersi_set, discodactylus_set, fuscifacies_set, conspicillatus_set, margaritifer_set
gc.collect()

# 5.) Image Pre-Processing
nRows = 543  # Width
nCols = 558  # Height
channels = 3  # Color Channels RGB-3, Grayscale-1

# 6.) Training and Testing Set Labeling
X_train = []
X_test = []
y_train = []
y_test = []

# 7.) Read and Label Each Image in the Training Set
for image in train_set:
    try:
        X_train.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
        if 'alfaroi' in image:
            y_train.append(1)
        elif 'cinerascens' in image:
            y_train.append(2)
        elif 'lanciformis' in image:
            y_train.append(3)
        elif 'bifurcus' in image:
            y_train.append(4)
        elif 'petersi' in image:
            y_train.append(5)
        elif 'discodactylus' in image:
            y_train.append(6)
        elif 'fuscifacies' in image:
            y_train.append(7)
        elif 'conspicillatus' in image:
            y_train.append(8)
        elif 'margaritifer' in image:
            y_train.append(9)
    except Exception:
        print('Failed to format: ', image)

print("Constructed training sets")

# 8.) Read and Label Each Image in the Testing Set
for image in test_set:
    try:
        X_test.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
        if 'alfaroi' in image:
            y_test.append(1)
        elif 'cinerascens' in image:
            y_test.append(2)
        elif 'lanciformis' in image:
            y_test.append(3)
        elif 'bifurcus' in image:
            y_test.append(4)
        elif 'petersi' in image:
            y_test.append(5)
        elif 'discodactylus' in image:
            y_test.append(6)
        elif 'fuscifacies' in image:
            y_test.append(7)
        elif 'conspicillatus' in image:
            y_test.append(8)
        elif 'margaritifer' in image:
            y_test.append(9)
    except Exception:
        print('Failed to format: ', image)

print("Constructed test sets")

# 9.) Garbage Collection
del train_set, test_set
gc.collect()

# 10.) Convert to Numpy Arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 11.) Switch Targets to Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Training on " + str(len(X_train)) + " samples, validating on " + str(len(X_test)) + " samples.")

#Standardize data
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen.fit(X_train)

train_iterator = datagen.flow(X_train, y_train, batch_size=16)
test_iterator = datagen.flow(X_test, y_test, batch_size=16)

print("Finished preprocessing images")

# 12.) Convolutional Neural Network

DROPOUT_RATE = 0.1

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(558, 543, 3), use_bias=True))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(DROPOUT_RATE))

model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(l=.01), use_bias=True))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(DROPOUT_RATE))

model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(l=.01), use_bias=True))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(DROPOUT_RATE))

model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(l=.01), use_bias=True))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(DROPOUT_RATE))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

print("Constructed model architecture")

# 13.) Model Summary
print(model.summary())

#14.) Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('regular_model_fixed_noise_four_sec_mel.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history = model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=30, validation_data=test_iterator, callbacks=[es, mc])
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, callbacks=[es, mc])
model.save("temp.h5")

#15.) Plot Accuracy Over Training Period
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#16.) Plot Loss Over Training Period
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
