import keras
import numpy as np, cv2, os
import tensorflow as tf
from tensorflow.keras import layers
import pathlib

train_dir = pathlib.Path("faces/train")
test_dir = pathlib.Path("faces/test")

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   train_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#예측을 위한 이미지 리사이징 (매우 중요)
image1 = cv2.imread("lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", cv2.IMREAD_COLOR)
image1 = cv2.resize(image1, (img_height, img_width))
image_array1 = tf.keras.preprocessing.image.img_to_array(image1)
image_array1 = tf.expand_dims(image_array1, 0)

class_names = train_ds.class_names
file_paths = train_ds.file_paths

num_classes = 4
backbone = tf.keras.applications.VGG16(include_top=False, input_shape=(img_height,img_width,3))
model = keras.Sequential()
backbone.trainable = False
model.add(backbone)
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes))

# model = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  # validation_data=val_ds,
  epochs=5
)

loss, accuracy = model.evaluate(test_ds)
print("accuracy : ", accuracy, "\n")
print("loss : ", loss)
# if accuracy >0.9: model.save("./model")
prediction1 = model.predict(image_array1)
score1 = tf.nn.softmax(prediction1[0])
print("prediction1 : ",prediction1)
print("score1 : ",score1)
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score1)], 100 * np.max(score1))
# )
