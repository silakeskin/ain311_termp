# ain311_termp
Hacettepe University Department of Artificial Intelligence Engineering , Fundamentals of Machine Learning Term Project. ChessMatic: Automated Move Recognition from Board Images

ChessMatic: Automated Move Recognition from Board Images

This project implements a YOLOv8-based object detection pipeline to identify and track the movement of chess pieces in sequential images. It combines advanced algorithms like the Hungarian Algorithm with OpenCV for efficient processing and visualization.

Table of Contents

Introduction

Features

Installation

Usage

Folder Structure

Introduction

The aim of this project is to analyze sequential images of a chessboard and detect the movement of chess pieces. It provides annotated output images highlighting the changes between frames and identifies which chess pieces have moved. This project is particularly useful for creating automated chess game analysis tools or for educational purposes. Original dataset link : https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset
Some augmentations are aplied on original dataset. Train data had 650 images. After augmentation, it has increased to 1450. Model file ("best.pt") contains the model trained with augmented data.
Features

Detection of chess pieces using YOLOv8.

Tracking and marking of moved pieces between two sequential images.

Visual representation of movements using bounding boxes and arrows.

Support for multiple chess piece classes, including both black and white pieces.

Configurable confidence thresholds for detection accuracy.

Installation

To run this project, follow the steps below:

Clone the repository:

git clone https://github.com/silakeskin/ain311_termp.git

Navigate to the project directory:

cd ain311_termp

Install the required Python dependencies:

pip install -r requirements.txt

Ensure that YOLOv8 is properly installed. You can find more information about installing YOLOv8 from the Ultralytics Documentation.

Usage

To use this project:

Place your input images in the test_selected folder or unzip the test_selected file which is provided in this repository. Ensure the images are named sequentially (e.g., 1.jpeg, 2.jpeg). The code will not work as it should for all the photos you will upload.

Unzip test_selected file

Run the main script:

python project_submit.py

The processed images with annotated movements will be saved in the marked folder.

Review the console output for detailed information about the detected and moved pieces.

Folder Structure

Below is the expected folder structure:

.
|-- best.pt                  # Trained YOLOv8 model
|-- project_submit.py        # Main script for processing images
|-- test_selected/           # Folder containing input images
|-- marked/                  # Folder for saving output images
|-- requirements.txt         # Python dependencies

!!! Training files are provided if you would like to check over. Please do not run training files ("project_yolov8_train" and "SSD_implementation")because it takes so much time and it can change the "best.pt"  file. !!!



