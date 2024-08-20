
# from flask import Flask, render_template, request, send_from_directory
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# import time

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model("my_model4.keras")

# # Define the upload folder for images
# app.config['UPLOAD_FOLDER'] = 'userimages'

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', prediction_text='No file part')

#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('index.html', prediction_text='No selected file')

#     if file:
#         # Save the uploaded image with a unique filename
#         img_filename = f"pred_{int(time.time())}.jpg"
#         img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
#         file.save(img_path)

#         # Preprocess the image for the model
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0  # Normalize pixel values

#         # Make prediction
#         prediction = model.predict(img_array)

#         # Display the result
#         result = "The image indicates the presence of a brain tumor." if prediction[0][0] > 0.5 else "This is a Healthy Brain."
#         return render_template('index.html', prediction_text=result, img_filename=img_filename)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)


















from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import time

from displayTumor import DisplayTumor


app = Flask(__name__)

# Load the pre-trained model
model = load_model("my_model4.keras")

# Define the upload folder for images
app.config['UPLOAD_FOLDER'] = 'userimages'

# Instantiate DisplayTumor object
display_tumor = DisplayTumor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file')

    if file:
        # Save the uploaded image with a unique filename
        img_filename = f"pred_{int(time.time())}.jpg"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        file.save(img_path)

        # Preprocess the image for the model
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Make prediction
        prediction = model.predict(img_array)

        # Display the result
        if prediction[0][0] > 0.5:
            result = "The image indicates the presence of a brain tumor."
            # Apply tumor highlighting
            display_tumor.readImage(cv2.imread(img_path))
            display_tumor.removeNoise()
            display_tumor.displayTumor()
            tumor_highlighted_img = display_tumor.getImage()
            # Save the highlighted image
            highlighted_img_filename = f"highlighted_{img_filename}"
            highlighted_img_path = os.path.join(app.config['UPLOAD_FOLDER'], highlighted_img_filename)
            cv2.imwrite(highlighted_img_path, tumor_highlighted_img)
        else:
            result = "This is a Healthy Brain."
            # No need to highlight the image if no tumor is present
            highlighted_img_filename = None

        return render_template('index.html', prediction_text=result, img_filename=img_filename, highlighted_img_filename=highlighted_img_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)





