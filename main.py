import os
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ELU
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join(os.curdir, "dataset/train")
test_dir = os.path.join(os.curdir, "dataset/test")


model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(160, 160, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(ELU())
model.add(Dense(128))
model.add(ELU())
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
try:
    model.load_weights('weights1.h5', by_name=True)
    print("reading weights done.")

finally:

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(160, 160),
        batch_size=64,
        class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = train_datagen.flow_from_directory(
        test_dir,
        target_size=(160, 160),
        batch_size=64,
        class_mode='binary')

    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=1e-3),
                  metrics=['acc'])

    model.fit_generator(
          train_gen,
          steps_per_epoch=191,
          epochs=30,
          validation_data=test_gen,
          validation_steps=67)

    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=1e-5),
                  metrics=['acc'])

    model.fit_generator(
          train_gen,
          steps_per_epoch=191,
          epochs=15,
          validation_data=test_gen,
          validation_steps=67)
    model.save_weights('weights1.h5')
