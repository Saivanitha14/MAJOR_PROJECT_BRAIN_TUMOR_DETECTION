import cv2
import numpy as np
from keras.models import model_from_json

# Load the trained model
def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_weights.h5")
    return loaded_model

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 128, 128, 1)
    return im2arr

# Function to predict using the loaded model
def predict_tumor(image_path, model):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    disease = ['No Tumor Detected', 'Tumor Detected']
    result = disease[class_idx]
    return result

# Provide the path to the MRI image you want to predict
image_path = r'brain_tumor_dataset\no\4 no.jpg'

# Load the trained model
loaded_model = load_model()

# Make a prediction
prediction = predict_tumor(image_path, loaded_model)
print("Prediction:", prediction)
