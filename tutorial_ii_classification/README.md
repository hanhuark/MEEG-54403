## Assignment 2 - Image Classification
Pool boiling is a heat transfer mechanism that dissipates a large amount of heat with minimal temperature increase by taking the advantage of the high latent heat of the working fluids. As such, boiling has been widely implemented in the thermal management of high-power-density systems, e.g., nuclear reactors, power electronics, and jet engines, among others. The critical heat flux (CHF) condition is a practical limit of pool boiling heat transfer. When CHF is triggered, the heater surface temperature ramps up rapidly (~ 150 C/min), leading to detrimental device failures. There is an increasing research interest in predicting CHF based on boiling images. <br/>

In this dataset, there are two folders, namely, “pre-CHF” and “post-CHF” that contain pool boiling images before and after CHF is triggered, respectively. The target of this problem is to develop a machine learning model to classify the boiling regime (pre or post CHF) based on boiling images. a. Split the data set into training, validation, and testing. This can be done before training with a separate package “split-folders” or directly in the code during training. b. Set up and train a model to classify the pre-CHF and post-CHF images. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model). Use EarlyStopping for fast convergence. c. Test the model using the reserved test data, report the confusion matrix, accuracy, precision, recall, F1 score, the receiver operating characteristic (ROC), and area under the curve (AUC). <br/>

The dataset for this assignment can be accessed at https://data.mendeley.com/datasets/5kjnphrbsz/1

**Tutorial**:<br>
[![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A_M5BRpdKr_I8H2Wy_4pkHCcsNuh7ae0?usp=sharing)

---

#### INTRODUCTION
Classification is a common machine learning task. It is used to predict the category an input belongs to based on its features. There are two types we will focus on; binary classification and multi-class classification. Binary classification describes the instance where there are only two possible categories while for multi-class there are multiple categories. This discussion will assume you have the background knowledge from the previous notes so if you haven't read those at least skim over them. <br><br>

Now for this discussion we will need to have a dataset in mind. We will uses images because that is what your homework assignment is about but keep in mind this could be done for any input datatype. An image consists of a set of pixel values that describe the color. For grey scale images each pixel has one value between 0 and 255. For colored images, each pixel has 3 values for the 3 seperate channels (rgb). Our dataset will be grey scale. 

To give you a better understanding, we will start with a small discussion using some image processing methods. Kernels (or filters) have been used for several applications in image processing. They are used for blurring/ smoothing images, sharpening images, or highlighting edges in images. All these kernels are applied to an image with convolution. Where a kernel is defined as a matrix of size mxn:

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

To apply the kernel to the image you first place it over a small portion of the image in this case we will do the top left corner. Then convolution is performed by multiplying each kernel value with the corresponding pixel value. All of these values are then summed to form a new pixel value. Then the kernel is moved across the image and this process is repeated to generate a new image.

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

Now if we pass these kernels over the image using convolution and fill out the new values we will get the following images. Notice how each one emphasize the edges horizontally and vertically. This kernel is used to create new versions of the same image that highlight specific features.

<img src="edge.png" alt="edgedetection" style="width:70%;">

This process is what CNN's are doing. However, we do not have defined kernels as we don't know what features we are looking for. So we set those with trainable parameters. That way the neural network and determine kernels that will result in the best model performance. 
