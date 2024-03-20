import copy
import numpy as np
# from client import *
from Federated.Server import *
from Model.RNN import RNN_Model
from Environment.Agent import Agent


class HFL(object):
    def __init__(self, NUM_USERS, LR, WEIGHT_DECAY_LAMBDA, NUM_ACTIONS, INPUT_SHAPE, NUM_TRAIN_TIME_FRAMES, FILE_NAME, CHECKPOINT_DIR, BATCH_SIZE):
        self.NUM_USERS = NUM_USERS
        self.LR = LR
        self.WEIGHT_DECAY_LAMBDA = WEIGHT_DECAY_LAMBDA
        self.INPUT_SHAPE = INPUT_SHAPE
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES
        self.FILE_NAME = FILE_NAME
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.BATCH_SIZE=BATCH_SIZE

    def create_client_server(self, dataset_train):
        num_items = int(len(dataset_train) / self.NUM_USERS)
        clients, each_client_idxs, all_idxs = [], [], [i for i in range(len(dataset_train))]
        net_glob = RNN_Model(
            LEARNING_RATE=self.LR, WEIGHT_DECAY_LAMBDA=self.WEIGHT_DECAY_LAMBDA, ACTION_SHAPE=self.NUM_ACTIONS, INPUT_SHAPE=self.BATCH_SIZE * self.INPUT_SHAPE, FILE_NAME=self.FILE_NAME + "_q_client", SAVE_CHECKPOINT_DIRECTORY=self.CHECKPOINT_DIR
        )

        # divide training data, i.i.d.
        # init models with same parameters
        for i in range(self.NUM_USERS):
            new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
            if(i == self.NUM_USERS - 1):
                new_idxs = set(all_idxs)
            
            all_idxs = list(set(all_idxs) - new_idxs)
            new_client = Agent(NUM_ACTIONS = self.NUM_ACTIONS, INPUT_SHAPE = self.INPUT_SHAPE, NUM_TRAIN_TIME_FRAMES = self.NUM_TRAIN_TIME_FRAMES, NAME = self.FILE_NAME, BATCH_SIZE=self.BATCH_SIZE)
            clients.append(new_client)
            each_client_idxs.append(new_idxs)

        server = Server(
            w=copy.deepcopy(net_glob.state_dict()), 
            LR=self.LR, WEIGHT_DECAY_LAMBDA=self.WEIGHT_DECAY_LAMBDA, NUM_ACTIONS=self.NUM_ACTIONS, INPUT_SHAPE=self.BATCH_SIZE * self.INPUT_SHAPE, FILE_NAME=self.FILE_NAME, CHECKPOINT_DIR=self.CHECKPOINT_DIR
        )

        return clients, server, each_client_idxs