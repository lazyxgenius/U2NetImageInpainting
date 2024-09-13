import os
import replicate
import requests
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

app = Flask(__name__)

# Set the directories
UPLOAD_FOLDER = r'C:\Users\Aditya PC\Desktop\u2netImageOutpainting\flask_uploads\images'
MASK_FOLDER = r'C:\Users\Aditya PC\Desktop\u2netImageOutpainting\flask_uploads\created_mask'
OUTPUT_FOLDER = r'C:\Users\Aditya PC\Desktop\u2netImageOutpainting\flask_uploads\output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to define the U-2-Net architecture
def U2Net(input_size=(320, 320, 3)):
    print("Building U2Net model...")
    inputs = Input(input_size)

    # Stage 1 - 4 of downsampling and upsampling
    h1 = RSU_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(h1)
    h2 = RSU_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(h2)
    h3 = RSU_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(h3)
    h4 = RSU_block(p3, 512)

    # UpSampling and Concatenation stages
    u3 = UpSampling2D((2, 2))(h4)
    u3 = Concatenate()([u3, h3])
    u3 = RSU_block(u3, 256, pool=False)

    u2 = UpSampling2D((2, 2))(u3)
    u2 = Concatenate()([u2, h2])
    u2 = RSU_block(u2, 128, pool=False)

    u1 = UpSampling2D((2, 2))(u2)
    u1 = Concatenate()([u1, h1])
    u1 = RSU_block(u1, 64, pool=False)

    # Final output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)

    model = Model(inputs=[inputs], outputs=[outputs])
    print("U2Net model built successfully!")
    return model


# Function to build the U-2-Net model
def RSU_block(x, filters, pool=True):
    print(f"RSU Block: Input shape: {x.shape}, filters: {filters}, pool: {pool}")

    # Initial Convolution block
    h = Conv2D(filters, (3, 3), padding='same')(x)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    # Downsampling and RSU block recursion
    if pool:
        p = MaxPooling2D((2, 2))(h)
        p = RSU_block(p, filters, pool=False)
        p = UpSampling2D((2, 2))(p)
    else:
        p = h

    # Upsampling and skip connections
    h = Conv2D(filters, (3, 3), padding='same')(p)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    return Concatenate()([x, h])

# Load U2Net weights (you can update the path)
model = U2Net()
model.load_weights(r"C:\Users\Aditya PC\Desktop\u2netImageOutpainting\u2net_3.weights.h5")

# Preprocess image for U2Net
def preprocess_image(image_path, target_size=(320, 320)):
    image = Image.open(image_path)
    original_size = image.size
    image = image.resize(target_size, Image.BILINEAR)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, original_size

# Postprocess mask from U2Net
def postprocess_mask(mask, original_size):
    mask = np.squeeze(mask)
    mask = (mask * 255).astype(np.uint8)
    mask = 255 - mask
    return Image.fromarray(mask).resize(original_size, Image.BILINEAR)

# Replicate inpainting function
def replicate_inpainting(source_image_path, mask_image_path):
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise EnvironmentError("Replicate API token not found.")

    with open(mask_image_path, "rb") as mask_file, open(source_image_path, "rb") as image_file:
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "mask": mask_file,
                "image": image_file,
                "width": 512,
                "height": 512,
                "prompt": "a wooden table",
                "scheduler": "DPMSolverMultistep",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 25
            }
        )

    output_url = output[0]
    response = requests.get(output_url)
    return response.content if response.status_code == 200 else None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Save uploaded image
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(image_path)

            # Generate mask using U2Net
            preprocessed_image, original_size = preprocess_image(image_path)
            predicted_mask = model.predict(preprocessed_image)
            mask_image = postprocess_mask(predicted_mask[0], original_size)
            mask_image_name = os.path.splitext(uploaded_file.filename)[0] + '_mask.png'
            mask_image_path = os.path.join(MASK_FOLDER, mask_image_name)
            mask_image.save(mask_image_path)

            # Perform inpainting using Replicate
            outpainted_content = replicate_inpainting(image_path, mask_image_path)
            if outpainted_content:
                outpainted_image_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(uploaded_file.filename)[0] + '_output.jpg')
                with open(outpainted_image_path, 'wb') as outpainted_file:
                    outpainted_file.write(outpainted_content)

                # Display original, mask, and output images
                return render_template('display.html',
                                       original_image=uploaded_file.filename,
                                       mask_image=mask_image_name,
                                       output_image=os.path.basename(outpainted_image_path))

    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Route to serve masks
@app.route('/masks/<filename>')
def mask_file(filename):
    return send_from_directory(MASK_FOLDER, filename)

# Route to serve output images
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
