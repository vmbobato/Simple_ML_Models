import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageOps

# Load the trained model
model = keras.models.load_model('mnist_model.h5')

# Function to load and preprocess an image
def load_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected.")
        return None

    # Open, convert to grayscale, resize, and normalize
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (white background, black digit)
    img = img.resize((28, 28))  # Resize to match MNIST input size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.reshape((1, 28, 28))  # Add batch dimension

    return img

# Load and predict the digit
image = load_image()
if image is not None:
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Show the image and prediction
    plt.imshow(image[0], cmap='gray')
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()

    print(f'Predicted Digit: {predicted_label}')
else:
    print("Image loading failed.")

