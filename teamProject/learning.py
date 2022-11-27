import numpy as np, cv2, os
import tensorflow as tf
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

class_names = train_ds.class_names
file_paths = train_ds.file_paths

num_classes = 3

model = tf.keras.Sequential([  # 학습모델 생성및 레이어 추가
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(  # 생성된 모델 묶기
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(  #모델 학습
  train_ds,
  # validation_data=val_ds,
  epochs=10
)

loss, accuracy = model.evaluate(test_ds)  #모델 검증(테스트)
print("accuracy : ", accuracy)
print("loss : ", loss)
if accuracy > 0.9: model.save("./model")  #조건 충족 시 학습된 모델 저장
