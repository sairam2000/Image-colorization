from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


def colorize(image_path):
    img_size = 120
    rgb_image = Image.open(
        image_path).resize((img_size, img_size))
    rgb_img_array = (np.asarray(rgb_image)) / 255
    gray_image = rgb_image.convert('L')
    gray_img_array = (np.asarray(gray_image).reshape(
        (img_size, img_size, 1))) / 255
    input_array = [gray_img_array]
    generator = load_model('./models/generator2.h5')
    y = generator(np.array(input_array)).numpy()
    image = Image.fromarray((y[0] * 255).astype('uint8')).resize((512, 512))
    image.save('./images/result.png')


colorize('./uploads/1.png')
