# Fast-Face-tracking Pipeline

This script provides a comprehensive pipeline for face detection, recognition, and classification in videos. It includes functionalities for extracting faces from videos, training machine learning models, training deep learning models, and running tests on trained models.

## Features

- **Face Extraction**: Extracts faces from video files using the UltraLight face detection model and saves them as individual images.
- **Clustering (Optional)**: Performs clustering on the extracted faces using KMeans clustering.
- **Machine Learning Models**: Trains various machine learning models (Logistic Regression, Random Forest, Decision Tree) on the extracted faces and evaluates their performance.
- **Deep Learning Models**: Trains Convolutional Neural Network (CNN) and Artificial Neural Network (ANN) models on the extracted faces using TensorFlow/Keras and evaluates their performance.
- **Face Detection and Recognition Test**: Runs tests on each trained model by applying it to frames of a video, detecting faces, and making predictions. Records results such as detection time and presence of the target face.
- **Combining Results**: Combines the results from the tests into a single DataFrame and saves it as a CSV file for further analysis.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- scikit-learn
- Seaborn
- Matplotlib

## Usage

1. **Extract Faces**: Prepare a video file and specify its path in the script. Run the script to extract faces from the video.

2. **Train Models**: Train machine learning models and deep learning models on the extracted faces by uncommenting the respective sections in the script.

3. **Run Tests**: Uncomment the test section in the script to run tests on trained models for face detection and recognition.

4. **View Results**: Check the generated CSV files containing the results of the tests for further analysis.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.
yamenmohamad@ gmail.com
