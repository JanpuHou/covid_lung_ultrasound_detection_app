from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def teachable_machine_classification(img, weights_file):
	model = load_model(weights_file)
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	image = img
    #image sizing
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
	image_array = np.asarray(image)
    # Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
	data[0] = normalized_image_array

    # run the inference
	prediction = model.predict(data)
	return np.argmax(prediction) # return position of the highest probability