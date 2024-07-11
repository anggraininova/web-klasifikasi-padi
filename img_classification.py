import keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predict the image class
    predictions = model.predict(data)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = round(100 * np.max(predictions[0]), 2)

    return predicted_class_index, confidence_score
