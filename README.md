# Breast Cancer Diagnosis using Convolutional Neural Networks

## Introduction
Hello there! Welcome to my Breast Cancer Diagnosis project using Convolutional Neural Networks (CNNs). This project is designed to classify breast cancer images into two categories: "Benign" and "Malignant." I've used TensorFlow and Keras to build a robust CNN model that can effectively analyze and categorize medical images.

## Dataset
The dataset used in this project is stored in the directory specified by the `datadir` variable. It contains images of breast tissue categorized as benign or malignant. The images are preprocessed and loaded into training, validation, and testing sets using TensorFlow's `image_dataset_from_directory` function.

## Model Architecture
The CNN model architecture is designed to extract relevant features from the input images. It consists of convolutional layers, max-pooling layers, and fully connected layers. The model is trained to learn patterns that differentiate between benign and malignant breast tissue.

## Data Augmentation
To enhance the model's robustness, I've applied data augmentation techniques using Keras's `Sequential` module. This includes rescaling, random rotation, random zoom, and random horizontal flipping. These techniques help prevent overfitting and improve the model's generalization capabilities.

## Training
The model is trained using the Adam optimizer with a learning rate schedule defined by an exponential decay function. I've implemented early stopping as a callback to monitor validation loss and prevent overfitting. The training process is visualized and can be observed in the output.

## Evaluation
I've included evaluation metrics to assess the model's performance on the training, validation, and testing datasets. The metrics include loss and accuracy. The final model summary is provided for a comprehensive overview of the architecture and parameters.

## Results
After training the model for a specified number of epochs (in this case, 20), I've saved the trained model as 'Breast-Cancer-Diagnosis-CNN.model'. You can use this saved model for making predictions on new breast cancer images.

Feel free to explore the code, modify hyperparameters, or use the trained model for your own applications. If you have any questions or suggestions, please don't hesitate to reach out. Happy coding!