from flask import Flask, render_template, request, redirect, url_for, after_this_request
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os
import threading

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model (make sure the path to your model is correct)
# model = tf.keras.models.load_model('./static/transfer_model2.keras')
model = tf.keras.models.load_model('./static/best_mobilenetv2_animals10.keras')

# Function to preprocess the image before passing to the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0   # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def delete_file_after_delay(filepath, delay=5):
    def delete_file():
        time.sleep(delay)
        try:
            os.remove(filepath)
            print(f"Deleted file: {filepath}")
        except Exception as e:
            print(f"Error deleting file: {e}")
    
    # Start a new thread to handle the delayed deletion
    threading.Thread(target=delete_file).start()


# Home route to upload images
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the 'static/uploads' directory
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        # Open the image and preprocess it
        image = Image.open(filepath)
        processed_image = preprocess_image(image)

        filename = os.path.basename(filepath)

        # Get the model's prediction
        prediction = model.predict(processed_image)

        # Get the index of the highest predicted class
        predicted_class = np.argmax(prediction[0])

        # Map predicted class to animal label (assuming you have a class_names list)
        class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
        result = class_names[predicted_class]

         # Delete the file after the response is sent to the client
        # @after_this_request
        # def remove_file(response):
        #     try:
        #         os.remove(filepath)  # Delete the image after rendering the results
        #     except Exception as e:
        #         print(f"Error deleting file: {e}")
        #     return response
        delete_file_after_delay(filepath, delay=5)


        return render_template('result.html', prediction=result, image_path=filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
