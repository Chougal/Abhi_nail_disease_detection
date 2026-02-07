# model.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input

input_shape = (224, 224, 3)

vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=input_shape)
)

# Freeze base model
for layer in vgg_base.layers:
    layer.trainable = False

x = Flatten()(vgg_base.output)

# ✅ MULTI-CLASS OUTPUT
output = Dense(17, activation="softmax")(x)

model = Model(inputs=vgg_base.input, outputs=output)
model.summary()
