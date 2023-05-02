from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import tensorflow as tf
import numpy as np

import time

app = Flask(__name__)

# Load the trained classification model
model = tf.keras.models.load_model('aeou_model.h5')

# Define the image size and channels for the input to the model
img_width, img_height = 28, 28
channels = 1

# Define the label encoder for one hot decoding
word_dict = {0: 'A', 1: 'E', 2: 'O', 3: 'U', 4:'Consonant'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    # Get the image data from the POST request
    img_data = request.values['imageBase64']

    # # Convert the base64 image data to a PIL Image object
    # img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes))
    
    # # Save the image as a PNG file
    
    # Preprocess the image for input to the model
    img = img.convert('L').resize((img_width, img_height))

    timestamp = time.time()
    img.save(f"./image/{timestamp}image.png")

    x = np.array(img).reshape(1, img_width, img_height, channels) / 255.0

    print(x.shape)
    
    # Run the classification model on the image
    y = model.predict(x)
    label = np.argmax(y)
    
    print(label)

    print(y)
    conf = int(np.max(y)*100)
    print(conf)

    # One hot decode the predicted label
    decoded_label = word_dict[label]
    
    # Return the predicted label as a JSON object
    return jsonify({'label': decoded_label, 'confident': conf})
    
if __name__ == '__main__':
    app.run(debug=True)
