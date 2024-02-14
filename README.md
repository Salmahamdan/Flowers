# Flowers
# Flower Image Classification using VGG16

## Overview
Implemented a deep learning model using TensorFlow and Keras to classify flower images from the Flower Photos dataset. Utilized the VGG16 pre-trained model as the base architecture and achieved a validation accuracy of approximately 86.65% after training for 10 epochs.

## Dataset
The Flower Photos dataset consists of images belonging to five different classes: daisy, dandelion, roses, sunflowers, and tulips. The dataset contains a total of 3670 images.

## Installation
To run the code locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flower-classification.git
   cd flower-classification
2. Install the required dependencies using pip:
       pip install -r requirements.txt
 3. Run the main script:
      python main.py
## Code Structure
- `main.py`: Main script to download the dataset, preprocess the images, define the model architecture, train the model, and evaluate its performance.
- `requirements.txt`: List of Python dependencies required to run the project.

## Model Architecture

 The model architecture consists of the following layers:

- VGG16 Base Model (pre-trained on ImageNet)
- Flatten Layer
- Dense Layer with ReLU activation
- Output Dense Layer with Softmax activation (5 classes)
  
## Training

  The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function. The training process involves 10 epochs with a batch size of 32.

## Evaluation

The trained model's performance is evaluated on a separate validation dataset, achieving a validation accuracy of approximately 86.65%.

## Results
- Training Accuracy: 99.97%
- Test Accuracy: 86.65%

## Conclusion

This project demonstrates the effectiveness of transfer learning using pre-trained models like VGG16 for image classification tasks. The trained model performs well in classifying flower images into different categories.

Feel free to experiment with different pre-trained models, optimizers, and hyperparameters to further improve the model's performance.
