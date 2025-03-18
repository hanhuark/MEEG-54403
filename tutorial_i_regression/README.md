### Assignment 1 - Regression:
In pool boiling experiments, the boiling heat flux can be estimated as the supplied power divided by the heater surface. However, this estimation will not be very accurate due to heat loss and other non-ideal conditions in experiments, especially for thin-film heaters with relatively low thermal conductivities (e.g., ITO heaters). Conventionally, finite-element simulations are used to evaluate the heat loss to validate or correct the experimental assumptions. Machine learning provides another perspective for tackling this issue. The heat loss and other non-ideal conditions can be captured and accounted for by the hidden layers of neural networks. The target of Problem 1-1 is to develop an MLP model to predict heat flux using temperature. The data set includes the temperature and the heat flux during a transient pool boiling test. a. Set up and train an MLP and a GPR model to predict the heat flux based on the temperature. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model). b. Circumvent the effects of overfitting using k-fold cross-validation (e.g., using 100 foldings).

---
#### INTRODUCTION
Regression is a supervised learning method that is used to predict continuous values. The following will explain some key concepts. We will start with a simple dataset. Our goal is to develop a model to predict y from x so we can approximate the value x=2.5.

Table 1: Dataset
|x|y|
|---|---|
|0|0|
|1|2|
|2|4|
|3|6|
|4|8|

Now, we don’t know the value of y for x=2.5. So it is important to determine if our model is good so we can trust our predictions. We can make an infinite number of models that can fit the data but they probably won’t all be good representations of it. That is why testing data is so important. We will want to withhold a random subset of our dataset from training so that we can run the model on it to understand how it did. In this case our testing set is chosen as:

Table 2: Testing Data

|x|y|
|--|--|
|1|2|
|4|8|

#### LINEAR REGRESSION

We will start with the simplest of these regression methods, linear regression. You have probably used this previously but for comparison lets go over the process behind it. The formula for ordinary least squares linear regression is:

Now we check our testing data:
And lastly we predict our value.
This can be implemented in python used the sklearn library:

#### NEURAL NETWORKS
The least squares linear regression method worked great for this simple dataset but with more complex non-linear datasets more complex models are needed to fully describe the translation from x to y. Deep learning methods describe models with multiple layers to extract features. The simplest of these is the multilayer perceptron. This model is inspired by the neurons in the human brain. The network is composed of multiple layers of neurons. Where one neuron, shown in figure NUMBER, describes a mathematical function. Specifically:

$$ y=\sigma \left( \sum W_ix_i+b \right)$$

Where $x_i$ are the inputs to the neuron, $W_i$ are the weights, b is a bias and sigma is an activation function. More on what all these mean later. To construct a mlp, you just specify neurons and layers. The activation functions can be used to add nonlinearity to the model or to specify the output type. For regression, the activation function of the last layer will be linear. This activation function is defined as (f(x)=x) or in other words the input remains unchanged. <br><br>
Say we have the simplest neural network we could create. It has one input, one neuron, one layer and the linear activation function.  The equation of the neuron will be y=Wx+b. <br><br>
Now we want to fit this model to our dataset so we can predict x=2.5. To do this we first randomly initialize all model weights and biases: Y=5*x+2 <br><br>
We then need to define a **loss function**. Our goal will be to minimize the loss function. We ideally want the loss to be zero implying that the prediction is the same as the label. For regression we will use mean squared error but there are several others that are commonly used such as mean absolute error.

$$ MSE= \frac{1}{n} \sum ^n_{i=1} \left( Y_i - Y_{pred,i} \right) ^2 $$

Where n is the number of data points, $Y_i$ is the true y and $Y_{pred,i}$ is the predicted value. <br><br>
So what we will do for training is pass our x inputs through our model to get predicted y values. 

|x|y|$y_{pred}$|
|--|--|--|
|0|0|2|
|2|4|12|
|3|6|17|

As you can see our predicted values are awful. This is okay because this is just a random guess. So now we will use these values to calculate our loss: MSE=63 <br><br>
Now we can use this loss value to update our weights and biases using an **optimizer** and **backpropagation** (which will be explained elsewhere). The most basic optimizer is *gradient descent* and is shown below: 

$$ W_x = W_x^* - \alpha \left( \frac{dLoss}{dW_x} \right) $$

$$ b_x = b_x^* - \alpha \left( \frac{dLoss}{db_x} \right) $$

Where $W_x$ and $b_x$ are the new weights and biases, $W_x^\*$ and $b_x^\*$ are the old weights and biases, and $\alpha$ is the learning rate. We will then take the derivative of the loss with respect to each weight and bias.

$$ \frac{dMSE}{dW} = \frac{1}{n} \sum_{i=1}^n 2 \left( Y_i - Y_{pred,i} \right) \left(-\frac{dY}{dW} \right) $$

$$ \frac{dMSE}{db} = \frac{1}{n} \sum_{i=1}^n 2 \left( Y_i - Y_{pred,i} \right) \left(-\frac{dY}{db} \right) $$

Where $\frac{dY}{dW} = x, \frac{dY}{db}=1$. <br><br>

Then we plug in all our values to solve for the derivatives of the loss function with respect to each learnable parameter. 

$$ \frac{dMSE}{dW}=32.67, \frac{dMSE}{db}=14 $$

Lets assume we are using a learning rate of 0.1. We can update our parameters as:

$$ W_{new}=1.733, b_{new}=0.6 $$


Now we have completed one full pass of the data. This is called an epoch. We will then continue our training process like this to better fit our model. We will continue until our training conditions are met. This could be many things such as: train for a set number of epochs, train until the loss is under a certain value, use validation data (more on this later) and train until it is under a certain loss, or you can set other more complicated stopping criteria. <br><br>

For this simple example, we will just specify the amount of epochs to complete. Note, we could get more accurate models by specifying the loss threshold and continuing training. Here we completed 10 epochs:

|Epoch|W|b|MSE|
|---|--|--|--|
|Initialize|5|2|63|
|1|1.733|0.6|0.13482|
|2|1.765|0.569|0.11735|
|3|1.779|0.5335|0.1032|

Figure NUMBER shows the training curve for the 10 epochs. The loss is decreasing over the training epochs which is what we want to see.<br><br>
We now will test our model with the withheld set. This will show if we trust our model to make predictions on data not used within the training. <br><br>
With our new W=1.859 and b=0.341 we will make predictions:

|x|y|$y_{pred}$|
|--|--|--|
|1|2|2.2|
|4|8|7.777|

Our testing MSE is 0.0449. This is similar to our training MSE loss. Whether or not this is acceptable is based on your own situation. For the case of this demo, we are going to say this error is fine. 
Finally, we can predict our y value for input x=2.5 using our model. We get 4.987. 

This same process can be done in python using Tensorflow: <br><br>


You may be asking “ this has been with a very simple network but what happens if we have more complicated networks?”
This is a great question. The process of upscaling is pretty simple now that we have this understanding. We will now get into backpropagation. Which is the process of updating the weights. It is composed of the following steps. These should look familiar to you since we just did a very simple example of them previously:

1.	Forward pass: make predictions
2.	Calculate loss 
3.	Backward pass: calculate partial derivatives of loss function via chain rule
4.	Optimizer: update model weights
   
Step 3 is where we will provide some more background on

#### OPTIMIZERS
*Gradient descent* uses all the data to the update the weights and biases. Although this leads to accurate and more stable updates the process requires high computation. Another method *Stochastic gradient descent* was proposed to fix this. It uses the same update equation but instead of using the entire dataset for updating, it uses a single point or smaller batches to update the gradients. This allows for faster training process but can lead to oscillations around the minimum. <r><br>

#### OVERFITTING

There are several things you have to look out for when training models. One important issue you need to avoid is overfitting. This can make it look like your training is going well but when checking on your testing data it has horrible performance. Overfitting is especially a danger with noisy data, you want to make sure your model describes trends rather than specific noise. 

#### VALIDATION DATA
Another important concept is validation data. I briefly mentioned it previously but said nothing beyond that. The inclusion of validation data is used for mitigating overfitting. This process works by setting aside a subset of training data. This data will be used at the end of an epoch to check in and see how well the training is going. The goal will be to minimize the loss of the validation data rather than the loss of the training data. 

#### GAUSSIAN PROCESS REGRESSION (GPR)


