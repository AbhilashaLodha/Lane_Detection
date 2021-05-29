# Lane_Detection
The project helps in detecting lanes on open source dataset namely "BDD100K Lane Marking Dataset" consisting of street images using deep learning model built with Keras on top of TensorFlow framework.

The steps involved in building up the project are -
1. Data Preprocessing 
2. Building a Fully Connected Neural Network
3. Model Training
4. Saving the model
5. Prediction on new test data

# Installations
Python 3.7.2 with a supporting IDE (Spyder, Jupyter Notebook etc.)

Install requisite libraries using below command -

pip install -r requirements.txt

# Dataset Used
The project is built up using open source dataset "BDD100K Lane Marking Dataset" that can be downloaded from https://s3.amazonaws.com/assignment-lane-marking-data/data.zip

The subset of this dataset contains around 500 images and their masks (labels) obtained by drawing the lane marking labels in such a way that everything else in the image but the lane markings are colored white.

After downloading the dataset, the images should be kept in "Images" folder and labels in "Labels" folder.

# Model Training
A fully connected deep learning model is built with Keras on top of TensorFlow framework. The model is trained for 50 epochs and we achieve train accuracy of around 89% and test accuracy of 85%. 

Once the model training gets completed, the trained model is saved in a json file (.json) along with its weights (.h5). Now this saved model can be used for doing the predictions on the new test data.

The command for training the model from scratch -

python model_training.py

# Model Predictions
The prediction file is a jupyter notebook file (lane_detection.ipynb) so that the results are clearly visible.

When the model is tested upon new data, it is able to detect lanes precisely and beautifully as shown in figures below -

Test image -
![1](https://user-images.githubusercontent.com/77407100/120079367-16506f00-c0d1-11eb-8b2c-b8449051173a.jpg)

Lane Detection -
![1](https://user-images.githubusercontent.com/77407100/120079384-22d4c780-c0d1-11eb-98c3-b8ccc3b60f3e.jpg)


