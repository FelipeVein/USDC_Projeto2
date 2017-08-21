# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imagens_para_writeup/imagem1.JPG "Data before"
[image2]: ./imagens_para_writeup/imagem2.JPG "Data after"
[image3]: ./imagens_para_writeup/imagem3.JPG "Before"
[image10]: ./imagens_para_writeup/imagem4.JPG "After"
[image4]: ./imagens_para_teste/50.jpg "Traffic Sign 1"
[image5]: ./imagens_para_teste/criancas.jpg "Traffic Sign 2"
[image6]: ./imagens_para_teste/criancas2.jpg "Traffic Sign 3"
[image7]: ./imagens_para_teste/curva_dir.jpg "Traffic Sign 4"
[image8]: ./imagens_para_teste/curva_esq.jpg "Traffic Sign 5"
[image9]: ./imagens_para_teste/pare.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Dataset Exploration


#### 1. The submission includes a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. The submission includes an exploratory visualization on the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was before (Fig 1) and after (Fig 2) I augmented the dataset. 

![alt text][image1]
![alt text][image2]



### Design and Test a Model Architecture

#### 1. The submission describes the preprocessing techniques used and why these techniques were chosen.

As a first step, I decided to augment the dataset because there are a lot of images of some signs, and few images of others signs. To do that, I used the images that I had, but changing them a little with cv2's GaussianBlur, cv2's equalizeHist and rotating the resulting image by a random angle. 

Here is an example of a traffic sign image before and after all this preprocessing techniques.

![alt text][image3]
![alt text][image10]


Then, I decided to pass all the images through a grayscale.

As a last step, I normalized the image data because the model learns better with zero mean and unit variance.



#### 2. The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 1 - 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU	1			|												|
| Max pooling 1	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 2 - 5x5	    | 1x1 stride, valid padding, outputs 10x10x16						|
| RELU	2				|												|
| Max pooling	2      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 3 - 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU	3				|												|
| Fully connected		| 800 inputs (relu3 + maxpool2) - 400 outputs         									|
| Dropout		| Keep Prob of 70%        									|
| Fully connected		| 400 inputs - 120 outputs          									|
| Dropout		| Keep Prob of 70%        									|
| Fully connected		| 120 inputs - 84 outputs          									|
| Dropout		| Keep Prob of 70%        									|
| Fully connected		| 84 inputs - 43 outputs        									|
| Softmax				| 43 outputs        									|
|						|												|
|						|												|
 


#### 3. The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

To train the model, I used an AdamOptimizer with batch size of 1024 (running on a GTX1060), 40 epochs, learning rate of 0.0009 and dropout of 0.7

#### 4. The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

My final model results were:
* training set accuracy of ~95%
* validation set accuracy of ~95%
* test set accuracy of 93.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

-- The first architecture chosen was the LeNet with coloured images, because I already had the pipeline for it. 

* What were some problems with the initial architecture?

-- The test accuracy was 90% at it's best.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

-- I read some papers in the field ([Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf); [Zhe Zhu](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)), but I decided to change the first architecture a bit. I never changed the activation functions, because ReLU is considered the best choice nowadays. 

-- I decided to add dropout layers because the model was overfitting sometimes, even with low epochs. 

* Which parameters were tuned? How were they adjusted and why?

-- The learning rate was decreased, because the model couldn't get 90%+ accuracy on the validation data.

-- Epoch was increased.

-- Batch size was increased to increase performance.

-- Dropout was incresead, due to the 3 dropout layers. When dropout was lower, the model almost couldn't learn anything.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

I've downloaded 6 brazillian's signs to see if my model would work in my country.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The last image might be difficult to classify because "STOP" is written in portuguese: "PARE". If the model had a RGB input, it could classify the sign by the solid red circle around the "PARE". 

#### 2. The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		| 30 km/h   									| 
| Children Crossing     			| Dangerous curve to the right 										|
| Children Crossing					| Double curve											|
| Keep right	      		| Turn left ahead					 				|
| Keep left			| No entry      							|
| Stop			| Keep left      							|


The model was able to correctly guess 0 of the 6 traffic signs, which gives an accuracy of 0%. To this dataset, the model was terrible. The best result I got with these 6 images was 33% (2 out of 6).

#### 3. The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Despite the 0% accuraccy, the coprrect label appears on all top 5 predictions, as we can see below.
50 km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .86         			| 30 km/h   									| 
| .12     				| *50 km/h*										|
| .006					| Go straight or left											|
| .0003	      			| 70 km/h					 				|
| .00004				    | Roundabout mandatory      							|

Children Crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .81         			| Dangerous curve to the right   									| 
| .14     				| *Children Crossing* 										|
| .024					| Pedestrians											|
| .0055	      			| Traffic Signals					 				|
| .0049				    | Keep Right      							|

Children Crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .36         			| Double Curve   									| 
| .26     				| Bicycles crossing 										|
| .138					| *Children Crossing*											|
| .135	      			| Beware of ice/snow					 				|
| .069				    | Road narrows to the right      							|

Keep right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42         			| Turn left ahead   									| 
| .37     				| End of all speed and passing limits 										|
| .18					| *Keep right*											|
| .0039	      			| End of no passing					 				|
| .0034				    | Go straight or right      							|

Keep left

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| No entry   									| 
| .13     				| *Keep left* 										|
| .094					| Turn right ahead											|
| .0085	      			| Go straight or left					 				|
| .0048				    | Priority road      							|

Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .83         			| Keep left   									| 
| .14     				| 70 km/h 										|
| .0073					| Go straight or left											|
| .0063	      			| *Stop*					 				|
| .0033				    | Turn right ahead      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


