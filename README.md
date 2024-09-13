# U2Net Image Outpainting with Flask & Replicate

This project demonstrates a Flask-based web application for image outpainting using a pre-trained U2Net model. The model is trained to generate masks for product images, and the **Replicate** API is used to outpaint (inpaint) the background of the product images. The application allows users to upload images, create masks, and generate outpainted images via a web interface.

## Project Structure

``` bash
U2NetImageOutpainting/
├── flask_uploads/                  # Stores uploaded images, generated masks, and final output images
│   ├── created_mask/               # Generated masks from the U2Net model
│   ├── images/                     # Uploaded images by users
│   ├── output/                     # Outpainted images (product + new background)
│
├── templates/                      # HTML templates for the web interface
│   ├── index.html                  # Upload page for the image
│   ├── display.html                # Display page showing original, mask, and outpainted image
│
├── testing_data/                   # Dataset used for training and testing
│   ├── created_mask/               # Mask images for testing
│   ├── images/                     # Test images
│   ├── output/                     # Output inpainted images for evaluation
│
├── app.py                          # Main Flask application file
├── main.ipynb                      # Jupyter Notebook detailing the training process
├── u2net_3.weights.h5              # Trained U2Net model weights
├── Image_bwmask_dataset.zip        # Dataset used for training U2Net (contains images and black/white masks)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
```

## Features
- **U2Net Model**: A pre-trained U2Net model is used to generate masks for the uploaded images. This model is trained on a custom dataset of product images and their corresponding black and white masks.
- **Flask Web Application**: The app allows users to upload images, generates masks automatically using the U2Net model, and then uses the **Replicate** API to fill in the background with the desired outpainting effect.
- **Replicate API Integration**: Background inpainting is done using the **Replicate** API, which applies Stable Diffusion-based background generation.

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- Flask
- TensorFlow (compatible with U2Net model)
- Replicate API
- Other dependencies listed in `requirements.txt`

You can install all necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Steps to Run the Project

1. **Download or Clone the Repository**

    ```bash
    git clone https://github.com/lazyxgenius/U2NetImageInpainting.git
    cd U2NetImageOutpainting
    ```

2. **Setting Up Replicate API Key**  
   You'll need an API key from Replicate for the inpainting step. Set up your environment variable for Replicate using the following:

    ```bash
    export REPLICATE_API_TOKEN="your_replicate_api_token"
    ```

3. **Prepare the Dataset**  
   The dataset (`Image_bwmask_dataset.zip`) used to train the U2Net model is provided. It contains the product images and black/white masks.

    * Unzip the dataset file:
    
    ```bash
    unzip Image_bwmask_dataset.zip
    ```

4. **Run the Flask Application**  
   To launch the Flask web application, simply run the following command:

    ```bash
    python app.py
    ```

   The app will be available on `http://127.0.0.1:5000/`. You can upload an image and see the original image, generated mask, and the final outpainted image.

5. **Training (Optional)**  
   If you wish to retrain the U2Net model, you can refer to `main.ipynb`. This notebook outlines the steps to train the U2Net model using the provided dataset on a GPU-enabled platform like Google Cloud.

6. **Testing Data**  
   The `testing_data` folder contains images, masks, and outputs generated during the development phase. You can use these files to verify your setup or for additional evaluation.

## File Descriptions

* **`app.py`**: The main Flask application that handles image uploads, mask generation, and background inpainting via Replicate.
* **`u2net_3.weights.h5`**: The pre-trained U2Net model used to generate masks for the images.
* **`main.ipynb`**: Jupyter notebook detailing how the U2Net model was trained using the custom dataset.
* **`flask_uploads/`**: Stores user-uploaded images, generated masks, and final output images.
* **`templates/`**: HTML files for rendering the web interface.

## Flask Application Workflow

1. **Upload an Image**: The user selects an image file from their computer and uploads it using the web interface.
2. **Mask Generation**: The uploaded image is processed by the U2Net model to generate a mask. This mask highlights the product in the image.
3. **Inpainting**: The mask and original image are sent to the Replicate API, which generates a new background for the image using Stable Diffusion.
4. **Display**: The web application displays the original image, generated mask, and final outpainted image.

