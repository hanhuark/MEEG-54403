## Topic 5 - Your Choice:

The goal of this homework assignment is to give you the opportunity to explore an area you take interest in.


---

### Option 1: Reinforcement Learning

**Tutorial**:<br> [![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11Qxu8kPlwHQBEB12R11LRJ6erQOk6FY1?usp=sharing) 

![rl](./static/rlwalkgif.gif)

Reinforcement learning is a type of machine learning that allows agents to learn with rewards and penalties. These models are often used for controlling robots. Q-Learning is a common and basic type of reinforcement learning and will be the type used in this assignment. 

To help you understand we will walk through a very simple example. Pretend we have the following maze. We want to figure out how to get from the start to the goal. It is super obvious but let's use reinforcement learning for it just to explain the concepts. We will use q-learning for training an agent in our maze environment. For q-learning, we will first need to define the possible states. For this case we will use each coordinate as a state. Now we need to define the possible actions. In this case at each state it has the ability to move up or down (except at the top and bottom). 


Now with this we can intialize our q-table. It should contain q values for ever possible action at every possible state. We will fill it with zeros as shwon below:

$$\begin{bmatrix}
State & Up & Down \\
0 & 0 & 0 \\
1 & 0 & 0 \\
2 & 0 & 0 \\
3 & 0 & 0 \\
\end{bmatrix}$$

A q-table describes how good an action given your current state. In theory, the best actions will be the one corresponding with the higher q values. Obviously our current table doesn't tell us any information about how good an action is. We will need to train our model to update the states. To do this we will assign rewards based on our goal. So we will give the agent 10 points for reaching state 3 and deduct 1 point for every move that isn't reaching state 3. We will update each q value with the following equation:

$$ Q(s,a) _{new} \leftarrow Q(s,a) _{old} + \alpha(r + \gamma \max _{a'} Q(s',a')- Q(s,a) _{old} )  $$

where $\alpha$ is the learning rate, r is the reward, $\gamma$ is the discount rate, $Q(s,a)$ is the q value for action a at state s. Let's initialize these values for now and go in more detail about them later. Lets set $\alpha=0.1,\gamma=0.99$. 

Now we will perform episodes, where the agent can explore the environment to update the q table with the goal of maximizing the total reward. We also need to specify when we say an epside ends. For this case, we will say it ends after taking a specified number of steps (let's use 6) or it reaches the goal. Let's walk through one complete episode. The agent will start at state 0. Now it can only make one action so it will move down. Since this isn't the goal, it gets a reward of -1. We will then update $Q(0,down)$ with the following:

$$Q(0,down) _{new} \leftarrow Q(0,down) _{old} + \alpha( r + max(Q(1,up),Q(1,down)) -Q(0,down))$$

$$Q(0,down)=0+0.1(-1+0-0)=-0.1$$

Now, we are state 1 and have taken 1 step. For this state we have two options of actions; up or down. For training we want to encourage exploring so a new term epsilon is introduced along with some randomness. For our example we will randomly pick a number between 0 and 1, if the number is higher than epsilon then we will take the action corresponding with the highest q value, if the number is lower than epsilon then we will randomly select a number. For this case, we will say epsilon constantly 0.9. (We will discuss decaying epsilon later). So, we randomly select 0.32 which is lower than epsilon so we will randomly select an action. For this case we pick up. So we move the agent to state 0 again and update the q value. 

$$Q(1,up) _{new} \leftarrow Q(1,up) _{old} + \alpha( r + max(Q(0,down)) -Q(1,up))= 0 + 0.1(-1-0.1-0)=-0.11$$

Now we have taken 2 steps and are at state 0. Now we can only go down so we will move to state 1. We will update our q value.

$$Q(0,down) _{new} \leftarrow Q(0,down) _{old} + \alpha( r + max(Q(1,up),Q(1,down)) -Q(0,down))=-0.1+0.1(-1+0+0.1)=-0.19$$

Now we have taken 3 steps and are at state 1. Since we have two possible options we will randomly select a number between 0 and 1. We pick 0.92 which is higher than epsilon. This means we will go with the action corresponding to the highest q value. In this case, that is the action down (0>-0.11). Now we update the q value. 

$$Q(1,down) _{new} \leftarrow Q(1,down) _{old} + \alpha( r + max(Q(2,up),Q(2,down)) -Q(1,down))= 0 + 0.1(-1-0-0)=-0.1$$

Now we are at state 2 and have taken 4 steps. So we have two possible actions; up or down. We randomly pick a number between 0 and 1. We get 0.21 which is less than epsilon. So we randomly select an action. In this case we pick down. This moves up to state 3 which is the goal so we get a reward of 10 and our episode ends after updating the q value. 

$$Q(2,down) _{new} \leftarrow Q(2,down) _{old} + \alpha( r + max(Q(3,up)) -Q(2,down))= 0 + 0.1(10-0-0)=1$$

After 1 episode our update q table looks like: 

$$\begin{bmatrix}
State & Up & Down \\
0 & X & -0.19 \\
1 & -0.11 & -.1 \\
2 & 0 & 1 \\
3 & 0 & X \\
\end{bmatrix}$$

Our total reward will be the sum of -1+-1+-1+10=7.

We will repeat the process and go through multiple episodes in effort to further improve this reward. To improve the efficiency of the training, you can also include a decaying epsilon. This entails decreasing the epsilon value throughout the training process so that the agent takes more moves corresponding to the higher q values rather than random selections. 

After training and developing the best Q table, it comes time to use the Q table. Pretend this is the Q table we got after training for multiple episodes:

$$\begin{bmatrix}
State & Up & Down \\
0 & X & -0.19 \\
1 & -0.11 & -.1 \\
2 & 0 & 1 \\
3 & 0 & X \\
\end{bmatrix}$$

We want to move through the maze based on our training. To do this we will begin at our starting point, state 1. We only have on action here so we will move down to state 1. Now we will take the action corresponding to the highest q value in this case down since (-.1 > -.11). So we will move to state 2. Now we will take the action with the highest q value in this case down since (1>0). Now we are at state 3 which is our goal. 

To summarize some key concepts:

The **environment** is place where the learning occurs for example a maze. 
An **agent** is what is moving around and interacts with the environment for example a robot.
A **state** is where the agent is. An **action** is what the agent can do for example move up or down**
**Rewards** are what an agent gets for making a move (positive if good and negative if bad). And the **Q-Table** is what stores values corresponding to how good an action is.


### Option 2: Generative Models

**Tutorial**:<br> [![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXDvPdT3ZjN9ax1FFKVB3pUVeOys6vLn?usp=sharing) 

#### Generative Adversarial Networks

GAN's are a type of neural networks. They are used for generative realistic synthetic data. And are used in image generation (e.g., deep fakes), style transfer, super-resolution, etc. 
The general idea is that there are two networks:

*  A **generator** that creates realistic fake data
*  A **discriminator** that tries to distinguish between real and the generated (fake) data (classification network) 

These two models are trained together and "fight" against each other. The generator wants to create such realistic data that the discriminator can't tell whats real and fake and the discriminator wants to be so good at distinguishing between real and fake images. 

The generator works by taking a random input and outputing the desired output shape. Several of these outputs and real data are then passed through the discriminator with the corresponding class labels (i.e., real or fake). Two custom loss functions are combined employed to improve the discriminator and generator. 

Discriminator loss:

$$
L_D = \text{CE}(1, D(x_{\text{real}})) + \text{CE}(0, D(x_{\text{fake}}))
$$

Generator loss:

$$
L_G = \text{CE}(1, D(x_{\text{fake}}))
$$


There are a couple of common challenges with training GANs such as it being hard to balance the generator and discriminator or the generator producing similar outputs. 

#### Diffusion Models

### Large Language Models (LLMs)
