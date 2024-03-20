# 1. DNN (Deep Neural Network)

- ```import Torch.nn```: to handle our layers when using Linear layer
- ```import torch.nn.functional```: for the value activation function for deep neural network
- ```import optim```: for optimizer

## Parameters:
- `Fc1`: is fully connected layers of the neural network. Fully connected layers are also known as dense layers. a fully connected layer (also called a dense layer) is a type of neural network layer where every neuron in the layer is connected to every neuron in the previous layer. These connections are represented by weights that are learned during training using backpropagation.
- `Linear`: is a PyTorch function that creates a linear transformation with the given input and output size. In other words, nn.Linear(input_size, output_size) creates a linear layer that takes an input tensor of size (batch_size, input_size) and outputs a tensor of size (batch_size, output_size).
- `Flatten`: is used to convert the input tensor from a 2D tensor of shape (batch_size, num_features) to a 1D tensor of shape (batch_size * num_features). This is necessary because the fully connected layers (fc1 and fc2) expect a 1D input tensor.
- `Relu`: is a rectified linear unit (ReLU) activation function. It is used to introduce non-linearity in the neural network. The rectified linear unit function is defined as max(0, x), which means that any negative value of x is replaced with 0, and any positive value of x is left unchanged. In PyTorch, the functional module contains various functions that can be used as activation functions in neural networks. In this case, functional.relu is applied to the output of fc1 to introduce non-linearity in the network.

## Functions:
- `Forward()`: Defines the forward pass of the neural network. The forward pass is the process of computing the output of the network given an input tensor. The forward() function takes an input tensor as an argument and returns an output tensor. The forward() function is called automatically when the network is applied to an input tensor, either during training or inference. It applies a series of linear and nonlinear transformations to the input tensor, typically implemented as layers of neurons, to compute a prediction or estimate for the target variable.


# 2. Agent:

## Parameters:
- `batch_size`: refers to the number of transitions (i.e., state, action, reward, next state) that are sampled from the agent's memory buffer and used to update the parameters at each training iteration. Using mini-batches of transitions instead of individual transitions can help reduce the variance of the updates and improve the stability of the learning process.
- `Memory`: refers to a buffer that stores the agent's past experiences (i.e., transitions) that it has observed during training. The memory buffer is typically implemented as a queue or deque, with a fixed maximum capacity. The agent samples transitions from this buffer randomly to update its network.
- `Gamma`: is a hyperparameter that controls the discount factor in the Bellman equation used to estimate the expected cumulative reward (or Q-value) of a state-action pair. The discount factor is used to give less weight to future rewards compared to immediate rewards, which can help the agent focus on immediate rewards and avoid being trapped in long-term suboptimal policies.
- `epsilon`: is a hyperparameter that controls the probability of the agent taking a random action (i.e., exploring) instead of choosing the action with the highest Q-value (i.e., exploiting). This is known as the epsilon-greedy policy.
- `epsilon_decay`: is a hyperparameter that determines how quickly the value of epsilon is reduced over time as the agent learns. The idea is to gradually decrease the probability of exploration and increase the probability of exploitation as the agent becomes more confident about its Q-value estimates.
- `epsilon_min`: is a hyperparameter that sets the minimum value of epsilon. This ensures that the agent continues to explore the state-action space even after it has converged to a good policy, to avoid getting stuck in local optima.
- `Lr`: learning rate

## Functions:
- `Forward`: is responsible for sampling a batch of transitions (i.e., state, action, reward, next state) from the agent's memory buffer and using them to update the Q-network parameters. The replay function takes a batch size argument, which determines the number of transitions to sample from memory for each training iteration. The batch of transitions is typically used to compute the loss function and backpropagate the gradients through the network to update its parameters.
- `Get_action`: is responsible for selecting an action based on the current state and the agent's policy. The policy can be either a deterministic policy that always selects the action with the highest Q-value or a stochastic policy that selects actions probabilistically based on the Q-values. The act function takes the current state as input and returns an action that the agent should take next. In some implementations, the act function can also incorporate an exploration strategy, such as epsilon-greedy, to encourage the agent to explore the state-action space and avoid getting stuck in local optima.


# 3. DRL (Deep Reinforncement Learning)

- `Episode`: refers to a complete sequence of interactions between an agent and its environment, starting from the initial state, and ending with a terminal state (either a goal or a failure state). The goal of the agent is to learn a policy that maximizes its expected cumulative reward over multiple episodes.