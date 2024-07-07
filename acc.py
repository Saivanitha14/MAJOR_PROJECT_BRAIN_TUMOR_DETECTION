import os
import numpy as np
import cv2
import pickle
from keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Global variables
X = []
Y = []
classifier = None
accuracy = None

def load_data(directory):
    X = []
    Y = []
    for label, class_name in enumerate(['no', 'yes']):
        class_directory = os.path.join(directory, class_name)
        for filename in os.listdir(class_directory):
            img = cv2.imread(os.path.join(class_directory, filename), 0)
            img = cv2.resize(img, (128, 128))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(128, 128, 1)
            X.append(im2arr)
            Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def main():
    global accuracy
    global classifier

    # Directory containing your image data
    dataset_directory = './brain_tumor_dataset/'

    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        X, Y = load_data(dataset_directory)
        np.save("Model/myimg_data.txt", X)
        np.save("Model/myimg_label.txt", Y)

    print(X.shape)
    print(Y.shape)

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Evaluate the model
        loss, accuracy = classifier.evaluate(X_test, to_categorical(Y_test))
        print("CNN Prediction Accuracy on Test Images: {:.2f}%".format(accuracy * 80))

if __name__ == "__main__":
    main()
