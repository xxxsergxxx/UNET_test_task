# Airbus Ship Detection Solution

This repository contains a solution for the Airbus Ship Detection challenge. The goal of the challenge is to detect ships in satellite images.

## Dataset

The dataset consists of satellite images along with corresponding ship annotations encoded in the Run Length Encoding (RLE) format. The dataset is provided in a CSV file named `train_ship_segmentations_v2.csv`, which contains information about the image filenames and their corresponding ship annotations.

## Data Preprocessing

1. **Adding `has_ship` Column**: The first step in data preprocessing is to determine whether each image contains a ship or not. This is done by adding a new column named `has_ship`, which is set to 1 if the `EncodedPixels` column is not empty (indicating the presence of a ship), and 0 otherwise.

2. **Balancing the Dataset**: To address class imbalance, the dataset is balanced by selecting a subset of images with ships and without ships. This helps improve the accuracy of the model.

3. **Image and Mask Cropping**: The images are cropped into 3x3 squares to reduce their size for model training. Additionally, the masks corresponding to ship annotations are decoded from the RLE format and cropped to match the cropped images.

## Model Architecture

The model architecture used for ship detection is a U-Net architecture, which is a convolutional neural network (CNN) commonly used for image segmentation tasks. The U-Net architecture consists of an encoder-decoder structure with skip connections to preserve spatial information.

## Training

1. **Data Generators**: Custom data generators are implemented to efficiently prepare and provide batches of data for model training. This is important for handling large datasets and avoiding memory issues.

2. **Loss Function**: The model is trained using a custom loss function, which is a combination of binary cross-entropy (BCE) loss and the Dice coefficient loss. This combination helps optimize the model for both pixel-wise accuracy and object detection.

3. **Model Checkpointing**: During training, model checkpoints are saved based on the validation loss to monitor the model's performance and prevent overfitting.

## Evaluation

The model's performance is evaluated using the Dice coefficient, which measures the overlap between the predicted masks and the ground truth masks. Additionally, visual comparisons are made between the original images, true masks, and predicted masks to assess the model's accuracy qualitatively.

## Results

The trained model achieves a Dice score of 0.62 on the validation set.

## Usage

To train the model, follow these steps:
1. Preprocess the dataset using the provided code.
2. Define the model architecture and compile it with the desired optimizer and loss function.
3. Train the model using the prepared data generators and monitor its performance.
4. Evaluate the trained model using the validation set and visualize the results.

## Requirements

- Python 3.x
- TensorFlow
- pandas
- scikit-learn
- scikit-image
- matplotlib

## Credits

This solution was developed by Kolotukhin Serhii as part of the Airbus Ship Detection challenge.

For more information, refer to the original challenge https://www.kaggle.com/c/airbus-ship-detection/overview.
