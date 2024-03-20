import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import os

class DNN(nn.Module):
    def __init__(self, LR, WEIGHT_DECAY_LAMBDA, NUM_ACTIONS, INPUT_SHAPE, NAME, CHECKPOINT_DIR,
                 fc1_dims = 512, fc2_dims = 512, fc3_dims = 512, fc4_dims = 512):
        super().__init__()
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, NAME)

        print("Input shape:", INPUT_SHAPE)
        print("num actions:", NUM_ACTIONS)

        self.INPUT_SHAPE = INPUT_SHAPE
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.NUM_ACTIONS = NUM_ACTIONS
        self.LR = LR
        self.WEIGHT_DECAY_LAMBDA = WEIGHT_DECAY_LAMBDA

        self.fc1 = nn.Linear(self.INPUT_SHAPE, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.NUM_ACTIONS)
        
        self.optimizer = opt.Adam(self.parameters(), lr=self.LR, amsgrad=True, weight_decay=self.WEIGHT_DECAY_LAMBDA)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        actions = self.fc5(x4)

        return actions
    
    def save_checkpoint(self):
        print(f'Saving {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'Loading {self.CHECKPOINT_FILE}...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))