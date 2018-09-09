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

[image1]: ./project_results/hist_initial_single_1.png  "Histogram of initial training data"
[image2]: ./project_results/multiHist_1.png "Histogram of training, validation and test data"
[image3]: ./project_results/dataExplore_1.png "Training data exploration"
[image4]: ./project_results/show_sign_and_count_color_1.png "Color signs and count"
[image5]: ./project_results/show_sign_and_count_gray_1.png "Gray signs and count"
[image6]: ./project_results/plot_image_label_image_1.png "Plot of images label 1"
[image7]: ./project_results/plot_image_flipX_1.png "Plot of images label 1 - augment/flip"
[image8]: ./project_results/plot_image_blurX_1.png "Plot of images label 1 - augment/blur"
[image9]: ./project_results/hist_augmented_1.png "Histogram of augmented data"
[image10]: ./project_results/sample_test_images.png "Test images"


CarND-Traffic-Sign-Classifier-Project

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The bar chart that shows number of samples per class. Around 8 classes have high representation (~2000 samples) while some just have 200 samples. 
The second image shows the training, validation and test data on the same plot
The third image prints some sample images


![Histogram of initial training data][image1]
![Histogram of training, validation and test data][image2]
![Training data exploration][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will reduce the depth of input from RGB to just Gray and will simplify the input and reduce complexity.

Here is an example of a traffic sign image before and after grayscaling. some features in dark images are only visible in gray scale
color is not giving any additional information. we are searching mostly for patterns. For example, images with labels 3, 19, 23, 38 are initially dark but they are visible in gray. 

After normalization, mean, std dev recenter etc, here is the pixel data
X_train[0]:  [[-0.84136963 -0.85649407 -0.85758454 ..., -0.79605383 -0.80899733   -0.80790681]
 [-0.82624525 -0.84567404 -0.84567404 ..., -0.79605383 -0.81008774   -0.81221122]
 [-0.82836866 -0.84567404 -0.84888804 ..., -0.79605383 -0.81008774   -0.81221122]
 
![Color signs and count][image4]
![Gray signs and count][image5]

As a last step, I normalized the image data because ...

I decided to generate additional data because some of the labels(classes) were under represented in the training set. To add more data to the the data set, I used two augmentation techniques.
1) flip the image horizontally to create a skewed viewing angle. Here if the initial image appears as pictured with camera on its left, it will be appear as if the camera was on the right of the image.
2) blur the image so that the quality is reduced.

There are also other techniques like rotating as well.

Here is an example of an original image and an augmented image:
![Plot of images label 1][image6]
![Plot of images label 1 - augment/flip][image7]
![Plot of images label 1 - augment/blur][image8]

The difference between the original data set and the augmented data set is the following ... 

Training sample count by class: Counter({2: 2010, 1: 1980, 13: 1920, 12: 1890, 38: 1860, 10: 1800, 4: 1770, 5: 1650, 25: 1350, 9: 1320, 7: 1290, 3: 1260, 8: 1260, 11: 1170, 35: 1080, 18: 1080, 17: 990, 31: 690, 14: 690, 33: 599, 26: 540, 15: 540, 28: 480, 23: 450, 30: 390, 16: 360, 34: 360, 6: 360, 36: 330, 22: 330, 40: 300, 20: 300, 39: 270, 21: 270, 29: 240, 24: 240, 41: 210, 42: 210, 32: 210, 27: 210, 37: 180, 19: 180, 0: 180})

Thefollowing classes have less samples. So I made the count equal to 1000 for each of them by augmentation and tripled the count.
 31: 690, 14: 690, 33: 599, 26: 540, 15: 540, 28: 480, 23: 450, 30: 390, 16: 360, 34: 360, 6: 360, 36: 330, 22: 330, 40: 300, 20: 300, 39: 270, 21: 270, 29: 240, 24: 240, 41: 210, 42: 210, 32: 210, 27: 210, 37: 180, 19: 180, 0: 180})
 
I followed the same process on the validation images as well.
Validation sample count by class: Counter({1: 240, 13: 240, 2: 240, 4: 210, 5: 210, 38: 210, 10: 210, 12: 210, 3: 150, 11: 150, 9: 150, 8: 150, 7: 150, 25: 150, 35: 120, 18: 120, 17: 120, 31: 90, 33: 90, 14: 90, 15: 90, 36: 60, 26: 60, 23: 60, 40: 60, 22: 60, 16: 60, 34: 60, 6: 60, 30: 60, 21: 60, 20: 60, 28: 60, 41: 30, 37: 30, 19: 30, 42: 30, 0: 30, 32: 30, 27: 30, 29: 30, 24: 30, 39: 30})

update only these samples
31: 90, 33: 90, 14: 90, 15: 90, 36: 60, 26: 60, 23: 60, 40: 60, 22: 60, 16: 60, 34: 60, 6: 60, 30: 60, 21: 60, 20: 60, 28: 60, 41: 30, 37: 30, 19: 30, 42: 30, 0: 30, 32: 30, 27: 30, 29: 30, 24: 30, 39: 30

I did not touch the test set. Below is the histogram plot after augmentation
 
![Histogram of augmented data][image9]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					|Description										| 
|:---------------------:|:-------------------------------------------------:| 
| Input					| 32x32x1 RGB image   								| 
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 28x28x6		| #(32-5+1)/1 = 28
| RELU					|													|
| Max pooling			| 2x2 stride,  , VALID padding outputs 14x14x6		|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 10x10x16		| #(14-5+1)/1 = 10
| RELU					|													|
| Max pooling			| 2x2 stride,  , VALID padding outputs 5x5x16		|
| Flatten				| Output = 400										|
| Fully connected		| Input = 400. Output = 120							|
| RELU					|													|
| Fully connected		| Input = 120. Output = 84							|
| RELU					|													|
| Dropout				| keep_prob = 0.5									|
| Fully connected		| Input = 84. Output = 43							|
|						|													|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, here are the hyper parameters I used:
EPOCHS = 100
BATCH_SIZE = 128 
optimizer - AdamOptimizer
learning rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy = 0.943
* test set accuracy = 0.9170229609392034
* training Loss = 0.013


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
- I started with the Lenet architecture from the lab.

* What were some problems with the initial architecture?
The validation accuracy never reached 0.93

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

From the Lenet architecture, max pooling was already implemented. I added dropout in the flattened and fully connected layers and iterated a few times and finally ended up using it between the last two fully connected layers, Layer 4 and 5. The reason for adjustment was overfitting on test data and underfitting on validation data as the validation accuracy was below 0.93.

* Which parameters were tuned? How were they adjusted and why?
I tried using a training rate of 0.005 but it did not help. Used dropout between layers 4 and 5 and it increased the validation accuracy to 0.943

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Lenet was a good starting point because it already uses convolution. Convolution reduces the number of input parameters and increases the speed to the calculation while retaining the accuracy. I added dropout to reduce overfitting on test data.

If a well known architecture was chosen:
* What architecture was chosen?
- Lenet architecture was chosen

* Why did you believe it would be relevant to the traffic sign application?
- It was good for the MNIST image data set in the lab and I also followed the project instructions to use it.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
- The test accuracy is below 0.93 which means that I may have to investigate the following
 * what image labels are causing a low test accuracy?
 * Are the images repsented in correct proportion in training, validation and test data sets?
 * Are the augmentation techniques used adding enough variety to the test and validation data

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![Test images][image10] 

The images were chosen so that they are present and represented in the test data set. I did not attempt to find images that may make the model fail.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image								| Prediction										| 
|:---------------------------------:|:-------------------------------------------------:| 
| Speed limit 20 km/h (0) 			| Speed limit 60 km/h (3) 							| 
| Speed limit 20 km/h (0) 			| Right-of-way at the next intersection (11) 		| 
| Keep Left (39) 					| Keep Left (39) 									| 
| Children Crossing (28) 			| Children Crossing (28) 							| 
| Yield (13) 						| Yield   (13) 										| 
| Roundabout Mandatory (40) 		| Roundabout Mandatory  (40) 						| 

The sings for "Speed limit 20 km/h" in training data set is different from the test data. The number font looks different and the test data was greyed out. It had a red border, test data had a blue border, the test data was in color. The two test images are also different. Once it predicted label ['3', 'Speed limit (60km/h)'] and other time it predicted label ['11', 'Right-of-way at the next intersection']



The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.66%. The accuracy on the test set was 0.91. These test images were in color and that may have impacted the accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


Top 5 softmax probabilies for the images
[[0.85 0.09 0.05 0 0 ]
[1 0 0 0 0 ]
[1 0 0 0 0 ]
[1 0 0 0 0 ]
[0.99 0 0 0 0]
[0.99 0 0 0 0]
]

Indices of the labels    
[ 1  5  3  2  8]
 [39 13 38 10  5]
 [28  0  1  2  3]
 [13  0  1  2  3]
 [40 12 10  7 42]
 [11  5 30  3  7]]
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


Reference:
https://www.datacamp.com/community/tutorials/tensorflow-tutorial
https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced


