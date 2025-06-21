from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from PIL import Image
import random # For unique filenames

# Import the modified DigitGenerator class
from your_model_file import DigitGenerator

# --- Configuration ---
# Define the path to your trained model weights.
# Make sure 'trained_generator_weights.h5' is in the same directory as app.py
# or provide the full path if it's elsewhere.
# If you leave this as None, an untrained model will be used.
# For deployment, ensure this file is included in your Git repo.
MODEL_WEIGHTS_PATH = 'trained_generator_weights.h5'

# Initialize the generator model globally when the app starts
# This ensures the model is loaded only once.
generator_model = DigitGenerator(weights_path=MODEL_WEIGHTS_PATH)
# If you remove the path: generator_model = DigitGenerator() # Will use untrained model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'generated_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to generate a digit image using your trained model
def generate_digit_image(digit):
    """
    Generates a single digit image using the loaded (or untrained) model.
    """
    try:
        # Pass target_digit if your model is a Conditional GAN, otherwise it's just for context
        pil_image = generator_model.generate_image_for_app(target_digit=digit)

        # Generate a unique filename to avoid caching issues and conflicts
        unique_filename = f'digit_{digit}_{random.randint(10000, 99999)}.png'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        pil_image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


@app.route('/')
def index():
    """
    Renders the main page of the web application.
    """
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_digits():
    """
    Handles the digit generation request from the front-end.
    """
    try:
        digit_str = request.form.get('digit')
        num_images_str = request.form.get('num_images')

        if not digit_str or not num_images_str:
            return jsonify({"error": "Digit and number of images are required."}), 400

        try:
            target_digit = int(digit_str)
            num_images = int(num_images_str)
            if not (0 <= target_digit <= 9):
                return jsonify({"error": "Digit must be between 0 and 9."}), 400
            if not (1 <= num_images <= 5):
                return jsonify({"error": "Number of images must be between 1 and 5."}), 400
        except ValueError:
            return jsonify({"error": "Invalid input for digit or number of images."}), 400

        generated_image_paths = []
        for i in range(num_images):
            image_path = generate_digit_image(target_digit)
            if image_path:
                generated_image_paths.append(os.path.basename(image_path))

        return jsonify({"images": generated_image_paths})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generated_images/<filename>')
def serve_image(filename):
    """
    Serves the generated images.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)