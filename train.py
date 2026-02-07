# train.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset/train/train"
test_dir = "dataset/test/test"

vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

for layer in vgg_base.layers:
    layer.trainable = False

x = Flatten()(vgg_base.output)

# ✅ 17 classes
output = Dense(17, activation="softmax")(x)

model = Model(vgg_base.input, output)

# ✅ categorical loss
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

model.fit(
    training_set,
    validation_data=test_set,
    epochs=10
)

model.save("model/nail_disease_model.h5")
print("✅ Model trained and saved")
