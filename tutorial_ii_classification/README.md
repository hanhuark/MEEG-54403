## Assignment 2 - Image Classification
Pool boiling is a heat transfer mechanism that dissipates a large amount of heat with minimal temperature increase by taking the advantage of the high latent heat of the working fluids. As such, boiling has been widely implemented in the thermal management of high-power-density systems, e.g., nuclear reactors, power electronics, and jet engines, among others. The critical heat flux (CHF) condition is a practical limit of pool boiling heat transfer. When CHF is triggered, the heater surface temperature ramps up rapidly (~ 150 C/min), leading to detrimental device failures. There is an increasing research interest in predicting CHF based on boiling images. <br/>

In this dataset, there are two folders, namely, “pre-CHF” and “post-CHF” that contain pool boiling images before and after CHF is triggered, respectively. The target of this problem is to develop a machine learning model to classify the boiling regime (pre or post CHF) based on boiling images. a. Split the data set into training, validation, and testing. This can be done before training with a separate package “split-folders” or directly in the code during training. b. Set up and train a model to classify the pre-CHF and post-CHF images. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model). Use EarlyStopping for fast convergence. c. Test the model using the reserved test data, report the confusion matrix, accuracy, precision, recall, F1 score, the receiver operating characteristic (ROC), and area under the curve (AUC). <br/>

The dataset for this assignment can be accessed at https://data.mendeley.com/datasets/5kjnphrbsz/1

**Tutorial**:<br>
[![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A_M5BRpdKr_I8H2Wy_4pkHCcsNuh7ae0?usp=sharing)

---

#### INTRODUCTION
Classification is a common machine learning task. It is used to predict the category an input belongs to based on its features. There are two types we will focus on; binary classification and multi-class classification. Binary classification describes the instance where there are only two possible categories while for multi-class there are multiple categories. This discussion will assume you have the background knowledge from the previous notes so if you haven't read those at least skim over them. <br><br>

Now for this discussion we will need to have a dataset in mind. We will uses images because that is what your homework assignment is about but keep in mind this could be done for any input datatype. 

To give you a better understanding, we will start with a small discussion using some image processing methods. Say I have an image that I want to do analysis on. An image consists of a set of pixel values that describe the color. For grey scale images each pixel has one value between 0 and 255. For colored images, each pixel has 3 values for the 3 seperate channels (rgb). As you can see this image is gray scale so we will only have one value per pixel. I want to figure emphasize the edges of the image. One way of doing this is to use a predefined kernel such as the Sobel kernels. Which are defined as:

$$ \begin{bmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1 \\
\end{bmatrix} , \begin{bmatrix}
+1 & +2 & +1 \\
0 & 0 & 0\\
-1 & -2 & -1 \end{bmatrix}$$

Now if we pass these kernels over the image using convolution and fill out the new values we will get the following images. Notice how each one emphasize the edges horizontally and vertically. This kernel is used to create new versions of the same image that highlight specific features.

This process is what CNN's are doing. However, we do not have defined kernels as we don't know what features we are looking for. So we set those with trainable parameters. That way the neural network and determine kernels that will result in the best model performance. 
