# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/0_new_image_4.jpg "New 20mph 1"
[image2]: ./resources/0_new_image_5.jpg "New 20mph 2"
[image3]: ./resources/0_new_image_53.jpg "New 20mph 3"
[image4]: ./resources/0_new_image_56.jpg "New 20mph 4"
[image5]: ./resources/0_new_image_85.jpg "New 20mph 5"
[image6]: ./resources/0_new_image_88.jpg "New 20mph 6"
[image7]: ./resources/0_original_image_29.jpg "Original 20mph" 
[image8]: ./resources/Hist_Test.png "Test Histogram"
[image9]: ./resources/Hist_Train.png "Training Histogram" 
[image10]: ./resources/Hist_Valid.png "Validation Histogram" 
[image11]: ./resources/confusion_matrix_best.png "Confusion Matrix"
[image12]: ./resources/webimages/20_speed_limit.jpg "Confusion Matrix"
[image13]: ./resources/webimages/general_caution.jpg "Confusion Matrix"
[image14]: ./resources/webimages/no_over_taking.jpg "Confusion Matrix"
[image15]: ./resources/webimages/priority_road.jpg "Confusion Matrix"
[image16]: ./resources/webimages/right_turn_ahead.jpg "Confusion Matrix"
[image17]: ./resources/webimages/stop.jpg "Confusion Matrix"
[image18]: ./resources/webimages/yield.jpg "Confusion Matrix"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nisheed75/SelfDrivingCar-CarND-TrafficSignClassifer/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

The Histogram below provide a visualization if the trainingg, validation and test data sets.  
##### The Train Data Histogram 
![alt text][image9]

##### The Validation Data Histogram 
![alt text][image10]

##### The Test Data Histogram 
![alt text][image8]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided not to convert the images to grayscale because because i wanrted to preserve the information in the color space.

I decided to generate additional data because some of my classes didn't have enough training examples so I generated images to have a good training data set with enough images in each class. To generate iamges i used the imgaug library and genreted images using this code:
```python
#Code taken from example in imageaug library https://github.com/aleju/imgaug
#____________________________________________________________________________________#

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

# Example batch of images.
# The array has shape (32, 32, 32, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(32, 32)) for _ in range(32)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips -- ******stopping this as it adding no value to the training data*****
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-15, 15),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

### End of borrowed Code _________________________________________________
```
Initally i was generating image and also flipping them but in my later training iterations i decided to remove the filliping as this was not adding value to the models accuracy. 

##### Original Image 
Here is an example of a traffic sign image before generating new images.

![alt text][image7]

##### Generated Images
Here is an example of generated images:
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

The difference between the original data set and the augmented data set is the following:
* Shape of training set x (34799, 32, 32, 3) y (34799,) before adding new images
* Shape of training set x (91827, 32, 32, 3) y (91827,) after adding new images

As a last step, I normalized the image data because data normlization helps to keep the training more efficient and helps with convergence. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I eneded up using the LeNet Architecture as this worked well for other images recognitions tasks and the training is more effiecinet so it takes less time to trian the model. I experimented with another home grown Architecture but this took 3X longer to train and yielded poorer results.

My final model consisted of the following layers:

| Layer         		|     Description	        					                        | 
|:---------------------:|:---------------------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							                        | 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 (Learns best color space)	|
| RELU					|												                        |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32	                        |
| RELU					|												                        |
| Dropout				|				With keep probability = 0.5							    |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x64	                        |
| RELU					|												                        |
| Max pooling	2x2     | 2x2 stride,  outputs 14x14x64 				                        |
| Dropout				|				With keep probability = 0.5								|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x128     						|
| RELU					|												                        |
| Max pooling	2x2     | 2x2 stride,  outputs 6x6x128 				                            |
| Dropout				|				With keep probability = 0.5								|
| Fully connected		| Inputs = 4608 (6x6x128)  Outputs = 1024       						|
| RELU					|												                        |
| Dropout				|				With keep probability = 0.5								|
| Fully connected		| Inputs = 1024 Outputs = 1024       									|
| RELU					|												                        |
| Dropout				|				With keep probability = 0.5								|
| Fully connected		| Inputs = 1024 Outputs = 512       									|
| RELU					|												                        |
| Dropout				|				With keep probability = 0.5								|
| Fully connected		| Inputs = 512 Outputs = 43 (Number of classes)       					|
| Softmax				| etc.        									                        |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an iterative approch as follows:
1. I defined a maximum number of EPOCHs for training,
1. For EPOCH i used a batch size  of 128
1. I used the SKLearn shuffle function to ensure i had random permutations for each EPOCH.
1. At the end of each EPOCH i checked the accuracy of the model.
1. I did this iteratively until the max EPOCH was reached.
1. At the end i save the model.
1.1. A i also took an itrative apprach to defining the model after we few experiments i decided to use early stopping and best model saving as i found that the best model isn't always your last EPOCH. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with LeNet becuase is performs well with image recognistion but i planned to iterarted and experiemnt with this to see what performace i could get with plain LeNet and then i tried a few steps to see if i could imporve on it. The process and changes I made are listed in the next section. 

##### Iterations and Changes Made
1. Iteration 1: I used plain LeNet with the training data set (with my generated images) this formed the basline for my experiments. I ran this for 40 EPOCHs to see what accuracy i could achieve.
1. Iteration 2: I ran this for 100 EPOCHs with evrything else unchanged and i got an improvement in accuracy. This was expected as the increased number of training runs gave the model more data to look at.
1. Iteration 3: I ran this for 150 EPOCHs with evrything else unchanged and i got a decline in accuracy. This was unexpected however i noticed that the best model was before the last EPOCH.
1. Iteration 4: I added L2 regularization to try and make sure i getting the lowesrt possible error (minimizing my error), added early stopping and best model saving so that i always got the best model not mater where is the lifecycle of EPOCH it occured. I also changed the learing rate to see if I could get more accuracy faster. My accuracy results were worst than the previous iterations.
1. Iteration 5: I changed the model architecture to be a custom 6 layer architecture but this was wasted effort as it took 4hrs (3.5x slower) to run and produced accuracy results well below previous results. 
1. Iteration 6: Went back to LeNet, took out the flipping that i was doing for image generation as i noticed this wasn't adding value to the validation accuracy. Fixed a code bug where i had an array size of nX64X64X3 and roated my imahes less. Running this got my accuracy back to the baseline performance.
1. Iteration 7: Using the exact data as Iteration 6 i changed the learning rate and i notced from Iteration 6 that i was overshooting so i changed the rate to 0.001. This gave me a significant improvement over the baseline. I stop here due to time constraints. 


##### Iteration Results and Parameters
|Iteration | Model                    |Batch Size|Learn Rate|Epoch Limit|Acutal EPOCHs|Training Accuracy|Validation Accuracy|
|---------:|:------------------------:|---------:|---------:|----------:|------------:|----------------:|------------------:|
|1         |LeNet                     |       128|     0.001|         40|           40|            0.964|              0.933|
|2         |LeNet                     |       128|     0.001|        100|          100|            0.983|              0.946|
|3         |LeNet                     |       128|     0.001|        150|          150|            0.947|              0.943|
|4         |LeNet                     |       128|     0.005|        100|           88|            0.913|              0.889|
|5         |ConVo (6 Layer)           |       128|     0.001|        100|          100|            0.774|              0.756|
|6         |LeNet                     |       128|     0.005|        100|          100|            0.942|              0.933|
|7         |LeNet                     |       128|     0.001|        100|           73|            0.990|              0.964|

##### Final
My final model results were:
Train Accurary = 0.990, Validation Accuracy = 0.9635, Test Accuracy = 0.9418

##### Confusion Matix 
The confusion Matrix below shows how the model performed against the validation data set.
![alt text][image11]

##### Summary
The model perfomed adequetly in triaing and validation but this isn't enough evidence that it is valid model as the traing process allows the model to use the training and validation data set to create the model. However when the model ran on the test data set (images is is seeing for the first time) the model still maintained a ~95% accuracy. This shows that the model is good enough to recognize traffic signs with low error.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image12] 
![alt text][image13] 
![alt text][image14] 
![alt text][image15] 
![alt text][image16]
![alt text][image17]
![alt text][image18]

The first image might be difficult as following 
1. The stop sign has a tree branch that is overlapping the sign boarder at the bottom.
1. The genral caution has poor contrast with the background 
1. The no passing, right turn ahead, stop, yield signs should be easy to classify as it very close to the training examples.
1. The priority road sign has some discolouration on the yellow part of the sign so this could cause classification issues.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| Speed limit (20km/h)							| 
| General caution   	| General caution  								|
| No Passing            | Np Passing                                    |
| Priority Road			| Priority Road									|
| Right Turn Ahead 		| Right Turn Ahead				 				|
| Stop					| Stop			      							|
| Yield					| Yield			      							|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.56%. These images are not that complex to predict so i was expecting a high accuracy with these images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 89th cell of the Ipython notebook.

##### Top softmax probabilities:
1. For the first image, the model was certain that this is a Speed limit (20km/h) sign (probability of 1.0)
20_speed_limit.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[0 1 2 3 4]

1. For the second image, the model was certain that this is a General caution sign (probability of 1.0)
general_caution.jpg:
Probabilities
[  1.00000000e+00   4.20215812e-34   0.00000000e+00   0.00000000e+00
   0.00000000e+00]
Corresponding labels
[18 27  0  1  2]


1. For the third image, the model was certain that this is a No Passing sign (probability of 1.0)
no_overtaking.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[9 0 1 2 3]

1. For the forth image, the model was certain that this is a Priority Road sign (probability of 1.0)
priority_road.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[12  0  1  2  3]

1. For the fifth image, the model was certain that this is a Right Turn Ahead sign (probability of 1.0)
right_turn_ahead.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[33  0  1  2  3]

1. For the sixth image, the model was certain that this is a Stop sign (probability of 1.0)
stop.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[14  0  1  2  3]

1. For the seventh image, the model was certain that this is a Yield sign (probability of 1.0)
yield.jpg:
Probabilities
[ 1.  0.  0.  0.  0.]
Corresponding labels
[13  0  1  2  3]

