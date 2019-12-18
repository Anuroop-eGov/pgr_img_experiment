from keras import Input
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Initialize standard VGG16 Model without FC layer
# Add custom Flatten and Dense layers to match output and input shapes
vgg_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(512, 512, 3)))
model = Sequential(name="VGG16_Preprocessor")
for layer in vgg_model.layers[:]:
    model.add(layer)
model.add(Flatten())
model.add(Dense(4096, input_shape=(131072,)))
print(model.summary())

# Loading test image
image = load_img('1001/test_7.jpg', target_size=(512, 512))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

print(model.predict(image)[0])
