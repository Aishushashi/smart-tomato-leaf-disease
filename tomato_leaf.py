from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pltimg_height = 224
img_width = 224
batch_size = 32train_dir = "data/train"
val_dir = "data/val"train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True
)val_datagen = ImageDataGenerator(
rescale=1./255
)train_gen = train_datagen.flow_from_directory(
train_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="categorical"
)val_gen = val_datagen.flow_from_directory(
val_dir,
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode="categorical"
)
