## Assignment 2 - Image Classification
Pool boiling is a heat transfer mechanism that dissipates a large amount of heat with minimal temperature increase by taking the advantage of the high latent heat of the working fluids. As such, boiling has been widely implemented in the thermal management of high-power-density systems, e.g., nuclear reactors, power electronics, and jet engines, among others. The critical heat flux (CHF) condition is a practical limit of pool boiling heat transfer. When CHF is triggered, the heater surface temperature ramps up rapidly (~ 150 C/min), leading to detrimental device failures. There is an increasing research interest in predicting CHF based on boiling images. <br/>

In this dataset, there are two folders, namely, “pre-CHF” and “post-CHF” that contain pool boiling images before and after CHF is triggered, respectively. The target of this problem is to develop a machine learning model to classify the boiling regime (pre or post CHF) based on boiling images. a. Split the data set into training, validation, and testing. This can be done before training with a separate package “split-folders” or directly in the code during training. b. Set up and train a model to classify the pre-CHF and post-CHF images. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model). Use EarlyStopping for fast convergence. c. Test the model using the reserved test data, report the confusion matrix, accuracy, precision, recall, F1 score, the receiver operating characteristic (ROC), and area under the curve (AUC). <br/>

The dataset for this assignment can be accessed at https://data.mendeley.com/datasets/5kjnphrbsz/1

**Tutorial**:<br>
[![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A_M5BRpdKr_I8H2Wy_4pkHCcsNuh7ae0?usp=sharing)

---

#### INTRODUCTION
Classification is a common machine learning task. It is used to predict the category an input belongs to based on its features. There are two types we will focus on; binary classification and multi-class classification. Binary classification describes the instance where there are only two possible categories while for multi-class there are multiple categories. This discussion will assume you have the background knowledge from the previous notes so if you haven't read those at least skim over them. <br><br>


#### CLASSIFICATION MODELS

Let's start with a very simple example dataset to learn the basic concepts of classification. Pretend we want to determine if inputs describe a cat or dog so we have the following dataset:

|weight|ear shape|tail length|animal|
|-----|---|----|---|
|4.0|pointed|25|cat|
|3.5|pointed|22|cat|
|30|floppy|35|dog|
|20|floppy|33|dog|

Once again you would set aside some data for testing your model but we will pretend that has already been done here so this is the training data. Next let's prepare our data. We will want numeric inputs so we need some way of handeling the ear shape and animal variables. We will need to perform categorical encoding. There are two main types; label encoding or one-hot encoding. For label encoding each category is assigned an integer. For one-hot encoding each value is converted to an array with 0's and 1's in columns corresponding to the category. Let's take a look at this visually: <br><br>
For label encoding: let's say pointed=0, floppy = 1 and cat=0, dog=1. So our training data would become:

|weight|ear shape|tail length|animal|
|-----|---|----|---|
|4.0|0|25|0|
|3.5|0|22|0|
|30|1|35|1|
|20|1|33|1|

Label encodings are simple and compact but they can imply an order. <br><br>
For one-hot encoding: let's say pointed is column 1 and floppy is column 2; cat is column 1 and dog is column 2. So our training data would become:

|weight|ear shape|tail length|animal|
|-----|---|----|---|
|4.0|[1,0]|25|[1,0]|
|3.5|[1,0]|22|[1,0]|
|30|[0,1]|35|[0,1]|
|20|[0,1]|33|[0,1]|

One-hot encodings are good because they don't imply an order and are espicially usefule for when an input belongs to multiple classes. However, they will take up more memory than the label encodings. <br><br>

For our case we will use the label encodings since this is just a simple binary classification case (we only have two possible output classes).
So what we will do is set up a neural network. This network will differ from the regression model because of the last layer. For binary classification, we will use one single output neuron with a sigmoid activation function which is defined below and looks like figure 

$$ \sigma (x)= \frac{1}{1+e^{-x}}$$

We will just set up the simplest network with only one layer and neuron which looks like this:

The output will be:

$$ output= \sigma(W_1 \cdot weight + W_2 \cdot earshape + W_3 \cdot taillength +b ) $$

So now due to the sigmoid function the output of our model will be between 0 and 1. So you may be wondering now how does a number between 0 and 1 tell me if it's a dog or cat? Well what you would do to tell what the prediction is choose a cut off (typically .5) and say if it's greater than this then it is classified as a dog and if it is less than this it is a cat. Let's define some random weights and biases and give you some examples of the prediction output. Let's say all the $W_i$'s are 1 and bias is 0. With this random assortment of weights we get the following outputs:

|model output|
|--|
|1|
|1|
|1|
|1|

Now we need to define a loss function. Be very careful here for classification. For this specific structure (binary classification, label encodings, one output neuron, and sigmoid activation function) we will want to use the binary crossentropy loss function:

$$BCE(y,\hat{y})= - [y\cdot log(\hat{y}) +(1-y)\cdot log(1-\hat{y})]$$

Now that we have a loss function this process is exactly the same as the regression neural network. We will just use back propogation to update all the weights iteratively in order to lower the loss. Then, once the model is trained we can perform classification. <br><br>

Let's talk a bit about multi-class classification. This is where we have more than two classes that the inputs could belong to. For example, predicting between cats, dogs, and foxes. For this instead of having one single output neuron there would be a number of neurons equal to the the number of classes. Additionally, two other things change; loss function and last layer activation function. For the last layer, we will now use a softmax activation function defined as:

$$\hat{y_i} = \frac{e^{Z_i}}{\sum^n_{j=1}e^{Z_j}} $$

This function just makes it so the outputs for one input sum up to one where n is the number of classes and $Z_i$ is the raw score for class i. 

The loss function used depends on the type of encodings you use for the labels. For label encodings you would use sparse categorical loss for one hot encodings you would use categorical cross entropy. Where categorical cross entropy is defined as:

$$ CCE(y,\hat{y}) = - \sum^n_{i=1} y_i log(\hat{y_i})$$

Now let's talk performance metrics. ROC, Confusion Matrix


#### CONVOLUTIONAL NEURAL NETWORKS
This talk has been about general classification but we will be using images. For images we need ways of extracting features to then predict the class. An image consists of a set of pixel values that describe the color or intensity. For grey scale images each pixel has one value between 0 and 255. For colored images, each pixel has 3 values for the 3 seperate channels (rgb). Our dataset will be grey scale. 

When it comes to neural network image analysis, CNN's are typically to way to go. CNN's are great at extracting features from images that are then used in predictions. To give you a better understanding, we will start with a small discussion using some image processing methods. Kernels (or filters) have been used for several applications in image processing. They are used for blurring/ smoothing images, sharpening images, or highlighting edges in images. Where a kernel is defined as a matrix of size mxn:

$$ K= \begin{bmatrix} 
W_{11} &.. & W_{1n} \\
: & : & : \\
W_{m1} & ..& W_{mn} \end{bmatrix} $$

And a gray scaled image is defined a matrix of size hxw:

$$I = \begin{bmatrix}
p_{11} & p_{12} & .. & p_{1w} \\
p_{21} & p_{22} & .. & p_{2w} \\
: &:&:&: \\
p_{h1} & p_{h2} & .. & p_{hw} \end{bmatrix}$$

Convolution is used to apply the kernel to the image. To do this, first place the kernel over a small portion of the image in this case we will do the top left corner. Then convolution is performed by multiplying each kernel value with the corresponding pixel value. All of these values are then summed to form a new pixel value. Then, the kernel is moved across the image and this process is repeated to generate a new image. The shift of the kernel is defined by a stride. 

$$ (I*K)(x,y)= \sum_{i=1}^m \sum_{j=1}^n I(x+i,y+j) \cdot K(i,j)$$

Okay now that we know how to use a kernel let's show a couple of examples. I want to emphasize the edges of the image so I will use a predefined kernel. There are quite a few but I will use the Sobel kernels. Which are defined as:

$$ \begin{bmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1 \\
\end{bmatrix} , \begin{bmatrix}
+1 & +2 & +1 \\
0 & 0 & 0\\
-1 & -2 & -1 \end{bmatrix}$$

Now if we pass these kernels over the image using convolution and fill out the new values we will get the following images. Notice how one emphasizes the edges horizontally and the other emphasizes the edges vertically. A kernel is used to create new versions of the same image that highlight specific features.

<img src="edge.png" alt="edgedetection" style="width:70%;">

What happens if you don't know what kernel to use for highlighting specific features or if you don't know what features are important for your application? This brings us to Convolutional Neural Networks. What we just walked through is the process used in CNN's however, they do not use predefined kernels. Instead kernels of specified sizes are initalized with trainable weights. That way the neural network can determine kernels that will result in the best model performance. For a convolutional layer the number of filters, size of the kernel and stride must be defined.<br><br>

So what we can do is pair these CNN layers with additional layers in a model for class prediction. The weights of these layers are updated with the rest of the layers during training. There are several layers that are commonly included such as: <br><br>

Max pooling

Flatten


