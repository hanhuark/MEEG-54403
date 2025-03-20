### Assignment 3 - Dimensionality Reduction and Clustering
Perform dimensionality reduction and clustering analysis of the same dataset used in HW-2.<br/>
(a) Run single value decomposition (SVD) or principal component analysis (PCA) of the images and plot the percentage explained variance vs. the number of principal components (PC).<br/> (b) Pick a representative image, run PCA and plot the reconstructed images using a different number of PCs (e.g. using PC1, PCs 1-2, PCs, 1-10, PCs 1-20, etc.).<br/> (c) Calculate the error of the reconstructed images relative to the original image and plot the error as a function of the number of PCs.<br/> (d) Run a clustering analysis of the boiling images using the PCs (the number of PCs to use is up to your choice) and evaluate the results of clustering.<br/>

The dataset for this assignment can be accessed at https://data.mendeley.com/datasets/5kjnphrbsz/1

---

#### INTRODUCTION
Unsupervised learning is commonly used for dimensionality reduction and clustering. This is unlike the work we have previously done since it does not use labeled data. Instead is works to find patterns within the data. For this section we will go other some popular unsupervised learning methods that you will implement. 

### PRINCIPLE COMPONENT ANALYSIS (PCA)
A very common dimensionality reduction algorthim is PCA. It works by defining a new set of basis vectors that capture the most variance. Now that may not make sense yet but please bear with me. As with most things it probably is easist to explain with an example. So let's take a very simple dataset:

|$x_1$|$x_2$|
|--|--|
|0|0|
|2|5|
|3|7|
|4|6|
|5|4|

Okay now pretend this huge dataset is way to large for your computer to handle or maybe you just want to eliminate noise. So you want to reduce the size of it. There are mulitple ways you could reduce the size for example maybe droping one of the $x_i$'s but if you do that you may be losing important information for what ever application you have planned for this data. So you decide PCA might be a great approach. I will walk you through the process then explain what it did to your data.

First you need to normalize the data by subtracting by the mean. In this case the mean is $[2.8,4.4]$.

|$x_1$|$x_2$|
|--|--|
|-2.8|-4.4|
|-0.8|0.6|
|0.2|2.6|
|1.2|1.6|
|2.2|-0.4|

Next you calculate the covariance matrix. This will describe how each variable varies in relation to the others. Since we have a 2D problem we will have a 2x2 matrix as follows:

$$
C = \begin{bmatrix}
\text{cov}(x_1, x_1) & \text{cov}(x_1, x_2) \\
\text{cov}(x_2, x_1) & \text{cov}(x_2, x_2)
\end{bmatrix}
$$

Where: 

$$cov(x_1,x_1)=\frac{1}{n-1}\sum_{i=1}^n(x_{1,i}^2)$$ 

$$cov(x_2,x_2)=\frac{1}{n-1}\sum_{i=1}^n(x_{2,i}^2)$$

$$cov(x_1,x_2)=cov(x_2,x_1)=\frac{1}{n-1}\sum_{i=1}^n(x_{1,i}*x_{2,i})$$

Plugging our values into these equations:

$$
C = \begin{bmatrix}
3.7 & 3.35 \\
3.35 & 7.3
\end{bmatrix}
$$

Now with the covariance matrix, you solve for the eigenvectors ($v$) and values ($\lambda$):

$$Cv=\lambda v$$

This is something you have probably seen before but I will quickly walk you through it:

$$det(C-\lambda I)=0$$

$$ det \left( \begin{bmatrix}
3.7-\lambda & 3.35 \\
3.35 & 7.3- \lambda 
\end{bmatrix} \right)= (3.7-\lambda)*(7.3-lambda)-(3.35)(3.35)=0$$

Then solving for $\lambda$ gives $\lambda = 1.697, \lambda = 9.303$. You then plug these back into the equation to get the corresponding eigenvectors $v=[-1.673,1] $ and $v=[0.598,1] $, respectively.
These eigen vectors now define our new space. Previously, our space was defined by the vectors [1,0] and [0,1]. Figure NUMBer shows the original basis and the new basis. You can actually represent every data point as a linear combination of the basis vectors. For example $[x1,x2]=[0.2,2.6]$

$$ \begin{bmatrix} 
0.2 \\ 
2.6 \end{bmatrix} = 0.2 \begin{bmatrix} 
1 \\
0 \end{bmatrix} + 2.6 \begin{bmatrix} 
0 \\
1 
\end{bmatrix} $$

$$ \begin{bmatrix} 
0.2 \\ 
2.6 \end{bmatrix} = 2.720 \begin{bmatrix} 
0.598 \\
1 \end{bmatrix} + 2.265 \begin{bmatrix} 
-1.673\\
1 
\end{bmatrix} $$

Now we sort the eigenvalues from highest to lowest. $9.303 > 1.697$ and use this to sort the corresponding vectors. 
The vectors with the highest corresponding vectors represent the directions with the most variance in the data. In our case the vector $v=[0.598,1]$ represents the most variance. 
