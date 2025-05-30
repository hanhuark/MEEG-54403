**For UofA students interested in enrolling in the machine learning class in Fall 2025:** If you cannot enroll in the course successfully, please use the following override forms:<br>
•	[ENGR Graduate Students](https://forms.office.com/pages/responsepage.aspx?id=xELHeRzmpU--iaPLVmqA0QFJWBwPcEBKrR3k9uL4ZXVUN1ZJTERQV0xZVUNMV0pQQ0JDSkZBV0FEMS4u&route=shorturl) <br>
•	[ENGR Undergraduate Students](https://forms.office.com/Pages/ResponsePage.aspx?id=xELHeRzmpU--iaPLVmqA0QFJWBwPcEBKrR3k9uL4ZXVUMU5SWUozWFZIWEhIN1NKS1hYQTJLSTlGMC4u) <br>
•	[Non-ENGR Students](https://forms.office.com/Pages/ResponsePage.aspx?id=xELHeRzmpU--iaPLVmqA0QFJWBwPcEBKrR3k9uL4ZXVUOUJNUldRMU1CUjY4MVI1SDQ2Q05NM1pOVC4u) <br>

# MEEG-44403/54403: Machine Learning for Mechanical Engineers
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=hanhuark/MEEG-54403)
## Instructor: [Han Hu](https://engineering.uark.edu/directory/index/uid/hanhu/name/Han+Hu/) and [Christy Dunlap](https://cldunlap73.github.io/)
## Organization: [Department of Mechanical Engineering, University of Arkansas](https://mechanical-engineering.uark.edu/)
## Course Description:
### Overview:
This course covers an introduction to supervised and unsupervised learning algorithms for engineering applications, such as visualization-based physical quantity predictions, dynamic signal classification, and prediction, data-driven control of dynamical systems, surrogate modeling, and dimensionality reduction, among others. The lectures cover the fundamental concepts and examples of developing machine learning models using Python and [MATLAB](https://www.mathworks.com/). This course includes four homework assignments to practice the application of different machine learning algorithms in specific mechanical engineering problems and a project assignment that gives the students the flexibility of selecting their topics to study using designated machine learning tools. The overarching goal of this project is to equip mechanical engineers with machine learning skills and deepen the integration of data science into the mechanical engineering curriculum. Compared to machine learning courses offered by computer science and data science programs, this course has a much stronger focus on integration with mechanical engineering problems. Students will be provided with concrete and specific engineering problems with experimental data. The projects, presentations, and in-class peer review practice are designed to foster students’ professional skills following the National Association of Colleges and Employers (NACE) competencies, including critical thinking, communication, teamwork, technology, leadership, and professionalism. Graduate students are required to complete an extra assignment (selected from three provided options) and a supercomputing assignment. 
<p align="center">
  <img src="images/MEEG-54403_Overview.png" alt="course_overview" width="100%" />
</p>

### Learning Objectives:
Students completing this course are expected to be capable of  <br>
•	Develop, train, and test machine learning models using Python/TensorFlow and MATLAB <br>
•	Develop machine learning models for image classification and clustering <br>
•	Perform data dimensionality reduction for physics extraction <br>
•	Analyze images/maps from experiments and simulations to predict physical quantities <br>
•	Adapt trained machine learning models to new applications <br>
•	Analyze time series for classification and regression <br>
•	Develop surrogate models for computationally expensive numerical simulations <br>
•	Benchmark the scalability of machine learning models on CPU and GPU clusters <br>
•	Develop complex machine learning models by integrating two or multiple mechanisms in tandem
### Textbook: 
This class does not require a textbook, but the following book is a good reference, particularly for Assignments 1 and 3. <br>
[Steven L. Brunton, J. Nathan Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control, 1st ed, Cambridge University Press, 2019](https://www.databookuw.com/)
### Software Packages:
Python Packages:
* TensorFlow <br>
* PyTorch <br>
* NumPy <br>
* SciPy <br>
* scikit-learn <br>
* Keras <br>
* Pandas <br>
* Matplotlib <br>
* Seaborn <br>
* OpenCV <br>

MATLAB and Toolboxes:
* [MATLAB](https://www.mathworks.com/products/matlab.html)
* [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
* [Statistics and Machine Learning Toolbox](https://www.mathworks.com/products/statistics.html)
* [Image Processing Toolbox](https://www.mathworks.com/products/image-processing.html)
* [System Identification Toolbox](https://www.mathworks.com/products/sysid.html)
### Assignment 1 - Regression:
In pool boiling experiments, the boiling heat flux can be estimated as the supplied power divided by the heater surface. However, this estimation will not be very accurate due to heat loss and other non-ideal conditions in experiments, especially for thin-film heaters with relatively low thermal conductivities (e.g., ITO heaters). Conventionally, finite-element simulations are used to evaluate the heat loss to validate or correct the experimental assumptions. Machine learning provides another perspective for tackling this issue. The heat loss and other non-ideal conditions can be captured and accounted for by the hidden layers of neural networks. The target of Problem 1-1 is to develop an MLP model to predict heat flux using temperature. The data set includes the temperature and the heat flux during a transient pool boiling test. 
a.	Set up and train an MLP and a GPR model to predict the heat flux based on the temperature. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model).
b.	Circumvent the effects of overfitting using k-fold cross-validation (e.g., using 100 foldings). 
The dataset for this assignment is provided in the folder. 

### Assignment 2 - Image Classification
Pool boiling is a heat transfer mechanism that dissipates a large amount of heat with minimal temperature increase by taking the advantage of the high latent heat of the working fluids. As such, boiling has been widely implemented in the thermal management of high-power-density systems, e.g., nuclear reactors, power electronics, and jet engines, among others. The critical heat flux (CHF) condition is a practical limit of pool boiling heat transfer. When CHF is triggered, the heater surface temperature ramps up rapidly (~ 150 C/min), leading to detrimental device failures. There is an increasing research interest in predicting CHF based on boiling images. This dataset includes two folders, namely, “pre-CHF” and “post-CHF” that contain pool boiling images before and after CHF is triggered, respectively. The target of this problem is to develop a machine learning model to classify the boiling regime (pre or post CHF) based on boiling images. 
a.	Split the data set into training, validation, and testing. This can be done before training with a separate package “split-folders” or directly in the code during training. 
b.	Set up and train a model to classify the pre-CHF and post-CHF images. Report the training curves (training/validation accuracy/loss vs. epoch) and the training time (time/epoch, time till the best model). Use EarlyStopping for fast convergence. 
c.	Test the model using the reserved test data, report the confusion matrix, accuracy, precision, recall, F1 score, the receiver operating characteristic (ROC), and area under the curve (AUC). 
The dataset for this assignment can be accessed at https://data.mendeley.com/datasets/5kjnphrbsz/1

### Assignment 3 - Dimensionality Reduction and Clustering
Run dimensionality reduction and clustering analysis of the same dataset used in HW-2.
(a)	Run single value decomposition (SVD) or principal component analysis (PCA) of the images and plot the percentage explained variance vs. the number of principal components (PC). 
(b)	Pick a representative image, run PCA and plot the reconstructed images using a different number of PCs (e.g. using PC1, PCs 1-2, PCs, 1-10, PCs 1-20, etc.).
(c)	Calculate the error of the reconstructed images relative to the original image and plot the error as a function of the number of PCs.
(d)	Run a clustering analysis of the boiling images using the PCs (the number of PCs to use is up to your choice) and evaluate the results of clustering. 
This assignment uses the same dataset as Assignment 2: https://data.mendeley.com/datasets/5kjnphrbsz/1

### Assignment 4 - Time Series Regression
The data file vapor_fraction.txt includes the vapor fraction (second column, dimensionless) vs. time (first column, unit: ms) of the boiling image sequences. The data are sampled with a frequency of 3,000 Hz (namely, a time step of 0.33 ms). Develop a recurrent neural network (RNN) model to forecast vapor fraction of future frames based on the past frames, e.g., predicting the vapor fraction profile of t = 33.33 ms – 66 ms using the vapor fraction history of t = 0.33 – 33 ms. Options include regular RNN, bidirectional RNN, gated recurrent unit (GRU), bidirectional GRU, long short-term memory (LSTM), bidirectional LSTM. 
(a)	Develop a baseline model with an input sequence length of 16.33 ms (50 data points) and an output sequence length of 16.33 ms (50 data points). Plot the model-predicted signal vs. the true signal. 
(b)	Vary the input and output sequence lengths to evaluate their effect on the error of the model predictions. 
The dataset for this assignment is provided in the folder. 

### Extra Assignment 1 - Image Classification using PCA-MLP
Re-do the image classification problem in HW-2 using PCA-MLP. Run SVD or PCA to obtain the PCs of the images. Feed the PCs to an MLP neural network to classify the regime of the boiling images. 
### Extra Assignment 2 - Image Regression
The vapor fraction (second) column of the data file vapor_fraction.txt are the labels of the images under the folder images Train a convolutional neural network (CNN) model to predict the vapor fraction of the images and compare the model prediction against the true data. 
### Extra Assignment 3 - Sequence to sequence prediction
The image dataset represents a boiling image sequence under transient heat loads. The images have a frame rate of 1,000 fps (or a time step of 1 ms). Run PCA to obtain the PC profiles of the image sequence. Feed the extracted PC profiles to an RNN model to forecast the PCs of future frames. Reconstruct image sequences using the predicted PCs and compare the reconstructed images against the true images. The recommended RNN models include LSTM or BiLSTM. 

Note: 
1.	Use an input vector length of 100 and an output vector length of 100 for the model.
2.	Downsample the image sequence (e.g., reducing the frame rate from 1,000 fps to 500 fps) in case of memory issues. 
## Tutorials for Assignments (MATLAB and Python):  
### Data source: [Nano Energy and Data-Driven Discovery Laboratory](https://ned3.uark.edu/) <br>
### Funding: [MathWorks Curriculum Development Support program](https://www.mathworks.com/company/aboutus/soc_mission/education.html) (Organized by [Mehdi Vahab](https://www.linkedin.com/in/mehdivahab/))
## Acknowledgments:
The initial development of the course was supported by the [Department of Mechanical Engineering at the University of Arkansas](https://mechanical-engineering.uark.edu/)(led by [Darin Nutter](https://mechanical-engineering.uark.edu/Directory/index/uid/dnutter/name/Darin-Nutter/)) and the [Arkansas NSF EPSCoR Data Analytics that are Robust and Trusted (DART) Project](https://dartproject.org/) (led by [Jennifer Fowler](https://www.arkansasedc.com/employee-detail/jennifer-fowler)). The experimental datasets used in the assignments were prepared by [Hari Pandey](https://www.linkedin.com/in/haripandey14/) and the development of the original course content was supported by [Connor Heo](https://www.linkedin.com/in/connor-heo-5b790336/) and [Christy Dunlap](https://www.linkedin.com/in/christy-dunlap/). The course syllabi and content were updated following the comments from the Department of Mechanical Engineering Curriculum Committee (chaired by [Steve Tung](https://mechanical-engineering.uark.edu/Directory/index/uid/chstung/name/Steve-Tung/)) and students enrolled in this course from 2021 - 2023. The Python tutorials for the assignments were developed by [Milad Sangsefidi](https://www.linkedin.com/in/milad-sangsefidi/) and [Daniel Peraza](https://www.linkedin.com/in/daniel-peraza-b65b171b9/) (Assignment 1), [Pengxiang Jiang](https://www.linkedin.com/in/pengxiang-jiang-b5b8515a/) (Assignment 2), [Mohammad Kokash](https://www.linkedin.com/in/mohammad-kokash/) (Assignment 3), and [Braden Stevens](https://www.linkedin.com/in/braden-stevens-pe/) (Assignment 4). The MATLAB tutorials for the assignments were developed by [Najee Stubbs](https://www.linkedin.com/in/najeei/) under the support of a gift from the [MathWorks Curriculum Development Support program](https://www.mathworks.com/company/aboutus/soc_mission/education.html) (organized by [Mehdi Vahab](https://www.linkedin.com/in/mehdivahab/)). 
## Publications:
### A. Publications consisting of course projects/assignments
* [J. K. Hoskins, H. Hu, and M. Zou, “Exploring Machine Learning and Machine Vision in Femtosecond Laser Machining,” ASME Open Journal of Engineering, vol. 2, p. 024501, 2023, doi: 10.1115/1.4063646.](https://asmedigitalcollection.asme.org/openengineering/article/doi/10.1115/1.4063646/1169944/Exploring-Machine-Learning-and-Machine-Vision-in) <br>
* [A.C. Iradukunda, B.M. Nafis, D. Huitink, Y. Chen, H.A. Mantooth, G. Campbell. and D. Underwood, "Toward Direct Cooling In High Voltage Power Electronics: Dielectric Fluid Microchannel Embedded Source Bussing Terminal," IEEE Transactions on Components, Packaging and Manufacturing Technology, 2024](https://ieeexplore.ieee.org/abstract/document/10443930) <br>
* [C. Dunlap, H. Pandey, and H. Hu, “Supervised and Unsupervised Learning Models for Detection of Critical Heat Flux during Pool Boiling,” in Proceedings of the ASME 2022 Heat Transfer Summer Conference, 2022, pp. HT2022-85582.](https://asmedigitalcollection.asme.org/HT/proceedings/HT2022/85796/V001T08A004/1146566) <br>
* [L. M. Jr, D. Jensen, and H. Hu, “Supporting Condition-Based Maintenance for Rotary Systems Under Multiple Fault Scenarios,” in Proceedings of the ASME 2023 International Design Engineering Technical Conferences & Computers and Information in Engineering (IDETC/CIE) Conference, 2023, p. V002T02A075.](https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings/IDETC-CIE2023/87295/V002T02A075/1170350?casa_token=-RofT6CbRZsAAAAA:CqTgXasH66LS3JHl5csGpWo0MxPkp4aXxJT5TVoFCNoE1F2e5-9x6aUy3Hx9JbZ5ZZO8n1us) <br>
* [C. Miller, "Generative Designs of Lightweight Air-Cooled Heat Exchangers," Mechanical Engineering Undergraduate Honors Thesis, University of Arkansas, May 2022](https://scholarworks.uark.edu/meeguht/111/) <br>
### B. Educational papers
* [H. Hu and C. Heo, “Integration of Data Science Into Thermal-Fluids Engineering Education,” in ASME International Mechanical Engineering Congress and Exposition, Proceedings (IMECE), 2022, vol. 7, no. Dl, pp. 1–10, doi: 10.1115/IMECE2022-88193](https://asmedigitalcollection.asme.org/IMECE/proceedings/IMECE2022/86694/V007T09A023/1157305) <br>
* [Y. Xu, B. Zhao, S. Tung, and H. Hu, “Infusing Data Science into Mechanical Engineering Curriculum with Course-Specific Machine Learning Modules,” in 2023 ASEE Annual Conference, Jun 2023, Baltimore, MD, 10.18260/1-2--43958.](https://peer.asee.org/infusing-data-science-into-mechanical-engineering-curriculum-with-course-specific-machine-learning-modules)
