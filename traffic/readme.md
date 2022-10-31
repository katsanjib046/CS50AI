# CS50AI: Traffic
## Description
This project is a part of the CS50AI course. In this project, we will use a variety of techniques to identify which traffic signs appear in a photograph. We will first write a function to load a dataset of images and their corresponding labels. Then, we will preprocess this data by resizing the images and encoding the labels. Next, we will design a convolutional neural network to classify the images. Finally, we will explore the effects of various parameters on the model’s accuracy. 
#### Note: Data sets haven't been uploaded to the repository. They are available in the course's website. Or Download using the following links:
* [Big set](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip)
* [Small set](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb-small.zip)
### First Attempt:
I have used the following model:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(43, activation="softmax")
])
```
 - Training Accuracy: 92.49%
 - Testing Accuracy: 97.43%

Big difference between training and testing accuracy.
### Second Attempt:
In the second attempt, I have added one additional CNN layer and removed the dropout. 
I have used the following model:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(43, activation="softmax")
])
```
 - Tranining Accuracy: 98.4%
 - Testing Accuracy: 96.84%

Smaller difference but still noticeable between training and testing accuracy.
### Third Attempt:
In the third attempt, I have set the dropout to 0.2. 
I have used the following model:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(30, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(43, activation="softmax")
])
```
 - Tranining Accuracy: 96.25%
 - Testing Accuracy: 96.72%

Training and testing accuracy are almost the same. Given that the accuracy is quite high and the difference between training and testing accuracy is small, I will use this model for the submission.