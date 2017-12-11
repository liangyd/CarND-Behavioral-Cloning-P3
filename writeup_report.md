# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/cnn.png 
[image2]: ./pictures/center_image.png 
[image3]: ./pictures/left_image.png 
[image4]: ./pictures/right_image.png 
[image5]: ./pictures/resize.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The ```model.py``` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been built

My model consists of a convolution neural network with 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. (model.py lines 99-125) This network architecture is based on Nvidia 's research [End to End Learning for Self-driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

This model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. For the loss function, I used the mean squared error. 

Here is a visualization of the architecture:

![alt text][image1]


#### 2. Attempts to reduce overfitting in the model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used the dropout technique (model.py lines 114). I added a dropout after the last convolution layer. I tried to put the dropout between other layers but the performance was not as good as this one.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 136). I tried to change the batch size and epoch to get a better result. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also drive the vehicle in both directions.

I collected 40643 images for the training process. There are three cameras on the vehicle. Here is an example image of center lane driving from the center camera:

![alt text][image2]

Here is an example image from the left camera:

![alt text][image3]

Here is an example image from the right camera:

![alt text][image4]


I used the images from all the three cameras. For the images in the left and right cameras, I add a correction angle of 0.25 to the original steering angle. 

The original image size is 160 x 320 x 3. The top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. Thus, I crop the top and bottom portions of an image. After that, I resized it to 66 x 200 x 3 to match the input image size of the Nvidia model. Here is an example of the cropped and resized image: 

![alt text][image5]

I also converted the color from BGR to YUV, which is required by the Nvidia model.

I used the python generator to generate the image data for training and validation rather than storing all the images in memory. I also shuffled the data.

#### 5. Training Process

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used the training data to train the model. The validation set helped determine if the model was over or under fitting. I chose a batch size of 128 and 10 epochs. I found that the model was overfitting, so I added a dropout technique to reduce overfitting. I also noticed that the validation loss did not become lower when I trained the model with more epochs. I also modified the batch size, and I noticed that a small batch size led to worse performance on track. 


### Simulation

After the training process with a batch size of 128 in 10 epochs, The model was saved in a file called model.h5. I started the simulation by executing 
```
python drive.py model.h5
```

The ```drive.py``` file uses the same image preprocessing strategy as ```model.py```. The only difference is that the color is changed from RGB to YUV in ```drive.py``` but it is changed from BGR to YUV in ```model.py```. The trained model predicts the steering command based on the image data input. 

When I opened the simulator and selected autonomous mode, the vehicle ran along the road automatically. I recorded a [video](https://youtu.be/JNt2j7QPjyw) which shows that the vehicle can run on track 1 autonomously.

