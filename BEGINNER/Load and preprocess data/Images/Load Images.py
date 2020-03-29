from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import logging

tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

data_root_orig = tf.keras.utils.get_file(
  origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
  fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
# print("data_root is:", data_root)

# print("subdir are:")
for item in data_root.iterdir():
  print(item)

import random

# 所有图片的路径
all_image_paths = list(data_root.glob('*/*'))
# 转换为列表
all_image_paths = [str(path) for path in all_image_paths]
# 随机化
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("image_count:", image_count)

print(all_image_paths[:10])

# 'daisy/7568630428_8cf0fc16ff_n.jpg CC-BY by A Guy Taking Pictures - https://www.flickr.com/photos/80901381@N04/7568630428/'
attributions = (data_root / "LICENSE.txt").open(encoding='utf-8').readlines()[4:]
# ['daisy/7568630428_8cf0fc16ff_n.jpg', ' by A Guy Taking Pictures - https://www.flickr.com/photos/80901381@N04/7568630428/\n']
# 列表
attributions = [line.split(' CC-BY') for line in attributions]
# 字典
# 'daisy/7568630428_8cf0fc16ff_n.jpg' -> ' by A Guy Taking Pictures - https://www.flickr.com/photos/80901381@N04/7568630428/'
attributions = dict(attributions)

import IPython.display as display


def caption_image(image_path):
  image_rel = pathlib.Path(image_path).relative_to(data_root)
  return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()
# 制作标签
# ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
# 编码标签
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
print(img_path)

img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100] + "...")

img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize(img_tensor, [192, 192])
# 归一化
img_final = img_final / 255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
print()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2, 2, n + 1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
  plt.show()

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False


# help(keras_applications.mobilenet_v2.preprocess_input)

def change_range(image, label):
  return 2 * image - 1, label


keras_ds = ds.map(change_range)

# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation='softmax')])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

len(model.trainable_variables)

model.summary()

steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()
print(steps_per_epoch)

model.fit(ds, epochs=1, steps_per_epoch=3)

import time

default_timeit_steps = 2 * steps_per_epoch + 1


def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # 在开始计时之前
  # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
  it = iter(ds.take(steps + 1))
  next(it)

  start = time.time()
  for i, (images, labels) in enumerate(it):
    if i % 10 == 0:
      print('.', end='')
  print()
  end = time.time()

  duration = end - start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))
  print("Total time: {}s".format(end - overall_start))


ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)

ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds

timeit(ds)

timeit(ds)

ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
ds

timeit(ds)

timeit(ds)

image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds

timeit(ds)

paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
image_ds

ds = image_ds.map(tf.io.serialize_tensor)
ds

tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)

ds = tf.data.TFRecordDataset('images.tfrec')


def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [192, 192, 3])
  return result


ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds

ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds

timeit(ds)
