import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from matplotlib import pyplot as plt

# Define the VGG16 model
vgg16_model = Sequential()
pretrained_vgg16 = tf.keras.applications.VGG16(include_top=False, input_shape=(64, 64, 3), pooling='avg', weights='imagenet')

train_dir = '/content/Train'
test_dir = '/content/Test'

for layer in pretrained_vgg16.layers:
    layer.trainable = False

vgg16_model.add(pretrained_vgg16)
vgg16_model.add(Flatten())
vgg16_model.add(Dense(512, activation='relu'))
vgg16_model.add(Dense(450, activation='relu'))
vgg16_model.add(Dense(260, activation='relu'))
vgg16_model.add(Dense(1, activation='sigmoid'))
vgg16_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the generators with the correct target size
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator_augmented = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=64,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=64,
    class_mode='binary'
)

# Calculate steps_per_epoch and validation_steps based on the size of your dataset
steps_per_epoch = train_generator_augmented.samples // train_generator_augmented.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

# Ensure steps_per_epoch and validation_steps are at least 1
steps_per_epoch = max(steps_per_epoch, 1)
validation_steps = max(validation_steps, 1)

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the VGG16 model
history_vgg16 = vgg16_model.fit(
    train_generator_augmented,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping],
    verbose=2
)


loss, accuracy = vgg16_model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss * 100:.2f}%")


# Plot training and validation accuracy
acc = history_vgg16.history['accuracy']
val_acc = history_vgg16.history['val_accuracy']
loss = history_vgg16.history['loss']
val_loss = history_vgg16.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the VGG16 model
vgg16_model.save('vgg16_model.h5')
vgg16_model.summary()