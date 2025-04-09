# Lung Cancer Image Classification Web Application

This repository contains a Flask web application that uses a deep learning model to classify lung cancer images into three categories:
- Lung Benign Tissue
- Lung Squamous Cell Carcinoma
- Lung Adenocarcinoma

## Project Overview

The application allows users to upload lung histopathology images through a web interface. The uploaded images are then processed by a pre-trained ResNet50-based deep learning model to predict the type of lung tissue or cancer present in the images.

## Features

- Drag and drop interface for image upload
- Support for multiple image uploads
- Real-time classification of uploaded images
- Display of classification results with corresponding images

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Deep Learning Framework**: TensorFlow/Keras
- **Model Architecture**: ResNet50 (pre-trained on ImageNet with custom top layers)

## Dataset

The model was trained on the `lung_image_set` dataset which contains histopathology images of:
- Lung benign tissue
- Lung squamous cell carcinoma
- Lung adenocarcinoma

### Downloading the Dataset

The dataset (~1GB) is not included in this repository due to size constraints. You can download it from Kaggle:

1. Visit the dataset page on Kaggle: [Lung Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download)
2. Click the "Download" button (Kaggle account required)
3. Extract the downloaded zip file to the project root directory
4. Ensure the extracted directory is named `lung_image_set` with the following structure:
   ```
   lung_image_set/
   ├── Lung_benign_tissue/
   ├── Lung_squamous_cell_carcinoma/
   └── Lung_adenocarcinoma/
   ```

Note: The dataset is only required if you want to retrain the model. For using the pre-trained model for inference, you only need the `Lung_cancer_prediction.keras` file.

## Project Structure

```
├── app.py                  # Flask application entry point
├── static/                 # Static files directory
│   ├── styles.css          # CSS styling
│   ├── script.js           # JavaScript for the frontend
│   └── uploads/            # Directory for uploaded images
├── templates/              # HTML templates
│   ├── index.html          # Upload page
│   └── results.html        # Results display page
├── Lung_cancer_prediction.keras  # Pre-trained model
├── Training_model.py       # Script used to train the model
├── requirements.txt        # Python dependencies
└── build.sh                # Build script for deployment
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/devendraBainda/lung-cancer-classifier.git
cd lung-cancer-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure you have the trained model file `Lung_cancer_prediction.keras` in the project root directory.

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to `http://127.0.0.1:5000/`

3. Upload lung histopathology images by either:
   - Dragging and dropping images onto the upload area
   - Clicking the upload area and selecting images from your file explorer

4. Click the "Predict" button to process the images

5. View the classification results for each uploaded image

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment Variables**: Add any necessary environment variables

Note: The pre-trained model file (`Lung_cancer_prediction.keras`) must be included in your repository or uploaded during the build process.

## Model Training

The model was trained on a dataset of lung histopathology images using transfer learning with a ResNet50 backbone. The training process included:

- Transfer learning using ResNet50 pre-trained on ImageNet
- Fine-tuning on lung histopathology images
- Data augmentation to improve model generalization
- Early stopping to prevent overfitting

The training code is available in `Training_model.py` for reference. If you wish to retrain the model, make sure to download the dataset first.

## Requirements

See `requirements.txt` for a full list of dependencies.


## Acknowledgements

- The ResNet50 model architecture was developed by Microsoft Research
- Dataset source: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download
