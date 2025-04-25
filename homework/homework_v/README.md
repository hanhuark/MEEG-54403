### Assignment 4 - Your Choice:

The goal of this homework assignment is to give you the opportunity to explore an area you take interest in.

The data file vapor_fraction.txt includes the vapor fraction (second column, dimensionless) vs. time (first column, unit: ms) of the boiling image sequences. The data are sampled with a frequency of 3,000 Hz (namely, a time step of 0.33 ms). Develop a recurrent neural network (RNN) model to forecast vapor fraction of future frames based on the past frames, e.g., predicting the vapor fraction profile of t = 33.33 ms – 66 ms using the vapor fraction history of t = 0.33 – 33 ms. Options include regular RNN, bidirectional RNN, gated recurrent unit (GRU), bidirectional GRU, long short-term memory (LSTM), bidirectional LSTM. (a) Develop a baseline model with an input sequence length of 16.33 ms (50 data points) and an output sequence length of 16.33 ms (50 data points). Plot the model-predicted signal vs. the true signal. (b) Vary the input and output sequence lengths to evaluate their effect on the error of the model predictions.

---

### Option 1: Reinforcement Learning

Reinforcement learning is a type of machine learning that allows agents to learn with rewards and penalties. These models are often used for controlling robots. Q-Learning is a common and basic type of reinforcement learning and will be the type used in this assignment. 

In general, q-learning works by allowing an agent to move around an environment and assigning values to each move based on goals you set. 

There are some key concepts you must understand:

The **environment** is place where the learning occurs for example a maze. 
An **agent** is what is moving around and interacts with the environment for example a robot.
A **state** is where the agent is. An **action** is what the agent can do for example move up or down**
**Rewards** are what an agent gets for making a move (positive if good and negative if bad). And the **Q-Table** is what stores values corresponding to how good an action is and will look like this to start with:

To show you an example pretend you have the following 

$$\begin{bmatrix}
State & Up & Down \\
0 & 0 & 0 \\
1 & 0 & 0 \\
2 & 0 & 0 \\
\end{bmatrix}$$

During training, you will update the q table. 

### Option 2: Generative 
