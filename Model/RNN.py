import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import math
import os

class RNN_Model(nn.Module):
    def __init__(self, LEARNING_RATE, WEIGHT_DECAY_LAMBDA, ACTION_SHAPE, INPUT_SHAPE, FILE_NAME, SAVE_CHECKPOINT_DIRECTORY,
                 H_LAYER1_DIMENSION, H_LAYER2_DIMENSION, H_LAYER3_DIMENSION, H_LAYER4_DIMENSION, H_LAYER5_DIMENSION, NUM_TRAIN_TIME_FRAMES, DEVICE):
        super().__init__()
        self.device = DEVICE

        # Saving parameters file
        self.checkpoint_dir     = SAVE_CHECKPOINT_DIRECTORY
        self.checkpoint_file    = os.path.join(self.checkpoint_dir, FILE_NAME)


        # Deep Neural Network layers dimensionnn.ReLU
        self.inp_dims   = INPUT_SHAPE
        self.ly1_dims   = H_LAYER1_DIMENSION
        self.ly2_dims   = H_LAYER2_DIMENSION
        self.ly3_dims   = H_LAYER3_DIMENSION
        self.ly4_dims   = H_LAYER4_DIMENSION
        self.ly5_dims   = H_LAYER5_DIMENSION
        self.trg_dims   = ACTION_SHAPE


        # Hyperparameter alpha (learning rate) and lambda (overfitting)
        self.lr         = LEARNING_RATE
        self.wght_lmbda = WEIGHT_DECAY_LAMBDA


        ''' Neural Network's layers
            (LSTM) --> 
            (CNN -> Batch Normalization -> MaxPool -> RELU) --> 
            (CNN -> Batch Normalization -> MaxPool -> RELU) -->
            (FC1 -> RELU) -->
            (FC2 -> RELU) -->
            (LINEAR DUELING)
        '''
        self.lstm       = nn.LSTM(self.inp_dims, self.ly1_dims, batch_first=True)

        self.conv1      = nn.Conv2d(1, self.ly2_dims, kernel_size=3) # one is for extra 4D dimension, layer1 dimension is considered in flatten
        self.bn1        = nn.BatchNorm2d(self.ly2_dims)
        self.relu1      = nn.ReLU()
        self.pool1      = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2      = nn.Conv2d(self.ly2_dims, self.ly3_dims, kernel_size=3)
        self.bn2        = nn.BatchNorm2d(self.ly3_dims)
        self.relu2      = nn.ReLU()
        self.pool2      = nn.MaxPool2d(kernel_size=2, stride=2)

        # All cnn with k=3 reduces dimension by -2 and all max_pool  wihth (k=2, s=2) reduces dimension by /2.
        self.flatten_reduction_width = math.floor(((math.floor((NUM_TRAIN_TIME_FRAMES - 2) / 2)) - 2) / 2)
        self.flatten_reduction_height= math.floor(((math.floor((self.ly1_dims - 2) / 2)) - 2) / 2)

        self.fc1        = nn.Linear(self.ly3_dims * self.flatten_reduction_width * self.flatten_reduction_height, self.ly4_dims)
        
        self.fc2        = nn.Linear(self.ly4_dims, self.ly5_dims)
        
        self.v_stream   = nn.Linear(self.ly5_dims, 1)
        self.a_stream   = nn.Linear(self.ly5_dims, self.trg_dims)

        # Initialize dueling with Xavier initialization and biases to zero
        nn.init.xavier_uniform_(self.v_stream.weight)
        nn.init.xavier_uniform_(self.a_stream.weight)
        nn.init.constant_(self.v_stream.bias, 0)
        nn.init.constant_(self.a_stream.bias, 0)

        # Using optimizer for time series while considering overfitting
        self.optimizer = opt.Adam(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.wght_lmbda)
        self.criterion = nn.MSELoss()

        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        # Input Layer
        if(len(state.shape) == 2):
            features, _ = self.lstm(state.unsqueeze(0))
        else:
            features, _ = self.lstm(state)
        x1 = F.leaky_relu(features).to(self.device)

        # permute for CNN
        x1 = x1.permute(0, 2, 1).unsqueeze(1).to(self.device)


        # Second Layer
        x2 = self.pool1(self.relu1(self.bn1(self.conv1(x1)))).to(self.device)

        # Third Layer
        x3 = self.pool2(self.relu2(self.bn2(self.conv2(x2)))).to(self.device)

        # Flatten for Linear
        x3 = x3.view(-1, self.ly3_dims * self.flatten_reduction_width * self.flatten_reduction_height) 

        # Fourth Layer
        x4 = F.leaky_relu(self.fc1(x3)).to(self.device)
        
        # Fifth Layer
        x5 = F.leaky_relu(self.fc2(x4)).to(self.device)
        
        # Output Layer
        V = self.v_stream(x5).to(self.device)
        A = self.a_stream(x5).to(self.device)
        actions = T.add(V, (A - A.mean(dim=1, keepdim=True))).to(self.device)

        if(len(state.shape) == 2):
            actions = actions[0]

        return actions
    
    def save_checkpoint(self):
        print(f'Saving {self.checkpoint_file}...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f'Loading {self.checkpoint_file}...')
        self.load_state_dict(T.load(self.checkpoint_file))
        