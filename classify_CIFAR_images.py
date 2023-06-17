import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Define the path to the downloaded dataset
dataset_path = "Datasets\cifar-10-batches-py"

# Load training data
train_images = []
train_labels = []
for i in range(1, 6):  # Assuming you have 5 training batch files
    with open(dataset_path + "/data_batch_" + str(i), "rb") as f:
        batch_data = pickle.load(f, encoding="bytes")
        train_images.append(batch_data[b"data"])
        train_labels.extend(batch_data[b"labels"])

# Load testing data
with open(dataset_path + "/test_batch", "rb") as f:
    test_data = pickle.load(f, encoding="bytes")
    test_images = test_data[b"data"]
    test_labels = test_data[b"labels"]

# Load label names from batches.meta
with open(dataset_path + "/batches.meta", "rb") as f:
    meta_data = pickle.load(f, encoding="bytes")
    label_names = meta_data[b"label_names"]

# Print the label names
print("Label names:", label_names)

# Convert lists to numpy arrays
train_images = np.concatenate(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Reshape the image data
train_images = train_images.reshape((train_images.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
test_images = test_images.reshape((test_images.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

# Print the shape of the loaded data
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Verify the data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Let's display the architecture of your model so far:
# model.summary()

# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Here's the complete architecture of your model:
model.summary()

# Compile  the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)