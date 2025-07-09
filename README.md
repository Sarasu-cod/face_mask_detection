# face mask detection
This project detects whether a person is wearing a face mask or not using computer vision and deep learning.

## Objective
Build and train a Convolutional Neural Network (CNN) model with TensorFlow and OpenCV to:
- Classify images as **With Mask** or **Without Mask**
- Perform real-time face mask detection through webcam

## Project Structure
face_mask_detection/
├── face_mask_detection.py # Main Python script
├── mask_detector.keras # Saved trained model (Keras format)
├── mask_detector_model.h5 # Saved trained model (HDF5 format)
├── Train/ # Training dataset (not uploaded to GitHub)
└── Validation/ # Validation dataset (not uploaded to GitHub)
> **Note:** The `Train` and `Validation` image folders are **mot included** in this repo due to large file size. Please download your own dataset and organize it in the same folder structure.

## Requirements
- Python
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scipy

## How to Run
1. Install dependencies:
   pip install tensorflow opencv-python numpy matplotlib scipy
2. Run the training & detection:
   python face_mask_detection.py
3. The webcam will open. The model will detect faces and predict mask status in real time.

## Skills Learned
- Image classification
- Object detection
- Model training and evaluation
- Real-time computer vision with OpenCV

## License
This project is for educational purposes only.

---

**Developed by:** Sara Susan Sunil
**Data:** July 2025
**Credits:** This project is part of my machine learning and computer vision learning journey.
