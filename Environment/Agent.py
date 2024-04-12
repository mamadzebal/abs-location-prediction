import torch as T
import numpy as np
from Model.RNN import RNN_Model
from Model.Memory import Memory
import random

rnd = np.random

class Agent(object):
    def __init__(self, NUM_ACTIONS, INPUT_SHAPE, NUM_TRAIN_TIME_FRAMES, NAME, ITERATION,
                # EPSILON=0, GAMMA=0.75, LR=0.001, WEIGHT_DECAY_LAMBDA=0, MEMORY_SIZE=1000000, BATCH_SIZE=10, EPSILON_MIN=0, EPSILON_DEC=5e-5, REPLACE_COUNTER=1000,
                EPSILON, EPSILON_MIN, EPSILON_DEC, GAMMA, LR, WEIGHT_DECAY_LAMBDA, BATCH_SIZE, MEMORY_SIZE, REPLACE_COUNTER,
                H_LAYER1_DIMENSION, H_LAYER2_DIMENSION, H_LAYER3_DIMENSION, H_LAYER4_DIMENSION, H_LAYER5_DIMENSION,
                NUM_TOP_SERVICES_SELECT, CHECKPOINT_DIR
                ):
        # Check if CUDA (GPU) is available
        # self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu" if T.cuda.is_available() else "cpu")
        self.ITERATION = ITERATION

        self.CHECKPOINT_DIR = CHECKPOINT_DIR

        self.loss = 0
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.lr = LR
        self.wght_lmbda = WEIGHT_DECAY_LAMBDA
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON_MIN = EPSILON_MIN
        self.EPSILON_DEC = EPSILON_DEC
        self.REPLACE_COUNTER = REPLACE_COUNTER
        self.MEMORY_SIZE = MEMORY_SIZE
        self.NUM_TOP_SERVICES_SELECT = NUM_TOP_SERVICES_SELECT

        self.num_actions = NUM_ACTIONS
        self.input_shape = INPUT_SHAPE
        self.NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES

        self.ACTION_SPACE = [i for i in range(self.num_actions)]
        self.learning_counter = 0

        self.memory = Memory(
            MAX_SIZE=MEMORY_SIZE, INPUT_SHAPE=NUM_TRAIN_TIME_FRAMES, NUM_ACTIONS=self.num_actions, NUM_TOP_SERVICES_SELECT=NUM_TOP_SERVICES_SELECT
        )
        self.q_online = RNN_Model(
            LEARNING_RATE=self.lr, WEIGHT_DECAY_LAMBDA=self.wght_lmbda, ACTION_SHAPE=self.num_actions, INPUT_SHAPE=self.BATCH_SIZE * self.input_shape, 
            H_LAYER1_DIMENSION=H_LAYER1_DIMENSION, H_LAYER2_DIMENSION=H_LAYER2_DIMENSION, H_LAYER3_DIMENSION=H_LAYER3_DIMENSION, H_LAYER4_DIMENSION=H_LAYER4_DIMENSION, H_LAYER5_DIMENSION=H_LAYER5_DIMENSION,
            FILE_NAME=NAME + "_q_online_abs_" + str(ITERATION), SAVE_CHECKPOINT_DIRECTORY=self.CHECKPOINT_DIR,
            NUM_TRAIN_TIME_FRAMES=NUM_TRAIN_TIME_FRAMES, DEVICE=self.device
        )
        self.q_future = RNN_Model(
            LEARNING_RATE=self.lr, WEIGHT_DECAY_LAMBDA=self.wght_lmbda, ACTION_SHAPE=self.num_actions, INPUT_SHAPE=self.BATCH_SIZE * self.input_shape, 
            H_LAYER1_DIMENSION=H_LAYER1_DIMENSION, H_LAYER2_DIMENSION=H_LAYER2_DIMENSION, H_LAYER3_DIMENSION=H_LAYER3_DIMENSION, H_LAYER4_DIMENSION=H_LAYER4_DIMENSION, H_LAYER5_DIMENSION=H_LAYER5_DIMENSION,
            FILE_NAME=NAME + "_q_future_abs_" + str(ITERATION), SAVE_CHECKPOINT_DIRECTORY=self.CHECKPOINT_DIR,
            NUM_TRAIN_TIME_FRAMES=NUM_TRAIN_TIME_FRAMES, DEVICE=self.device
        )

        # Move models to GPU
        self.q_online.to(self.device)
        self.q_future.to(self.device)


    def store_transition(self, state, action, reward, resulted_state, done):
        self.memory.store_transition(state, action, reward, resulted_state, done)
        
    def sample_memory(self):
        states, actions, rewards, resulted_states, dones = self.memory.sample_buffer(self.BATCH_SIZE)
        states = T.tensor(states)
        rewards = T.tensor(rewards)
        actions = T.tensor(actions)
        resulted_states = T.tensor(resulted_states)
        dones = T.tensor(dones)

        return states, actions, rewards, resulted_states, dones
    



    def get_action(self, flat_tensor):
        topk_values, topk_indices = T.topk(flat_tensor, k=self.NUM_TOP_SERVICES_SELECT)
        
        return topk_indices.numpy()
    
    def get_predicted_services_from_learning(self, state, train_mode):
        state = T.tensor(state, dtype=T.float)

        # Set to train in order to run dropout and batch normalization
        if(train_mode):
            self.q_online.train()

        expected_values = self.q_online.forward(state)
        actions = self.get_action(expected_values)

        return actions

    def get_top_predicted_services(self, state, SEED, train_mode = True):
        rnd.seed(SEED)

        # target action should be an array existing [0-NUM_TOP_SERVICES_SELECT] of top NUM_TOP_SERVICES_SELECT with probability to exist on next time frame
        if train_mode:
            random_number = rnd.random()
            if random_number > self.EPSILON:
                actions = self.get_predicted_services_from_learning(state, train_mode)
            else:
                actions = random.sample(range(self.num_actions), self.NUM_TOP_SERVICES_SELECT)
        else:
            actions = self.get_predicted_services_from_learning(state, train_mode)

        return actions

   
    def replace_target_network(self):
        if self.learning_counter % self.REPLACE_COUNTER == 0:
            self.q_future.load_state_dict(self.q_online.state_dict())

    def decrement_epsilon(self):
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON = self.EPSILON - self.EPSILON_DEC
        else:
            self.EPSILON = self.EPSILON_MIN


    def learn(self):
        if self.memory.counter < self.BATCH_SIZE:
            return 'empty'
        
        '''
        Using Double Deep Q-network (expected_value, action)
            self.q_online: select max_actions
            self.q_future: predict future
        '''

        
        # empty both NN models gradients
        self.q_online.optimizer.zero_grad()
        # self.q_future.optimizer.zero_grad()

        # periodically hard update q_future with q_online every REPLACE_COUNTER
        #  it helps to mitigate issues related to the non-stationarity of the targets by reducing the correlation between the target and predicted Q-values
        self.replace_target_network()

        # Experience Replay: randomly select states from memory (0 - BATCH_SIZE) in order to not let one situation impact a lot on model
        states, actions, rewards, resulted_states, dones = self.sample_memory()

        
        # Perform forward pass with q_online
        q_online_pred = self.q_online.forward(states)
        
        # extract the values from q_online_pred using the indices specified in actions
       
        q_online_pred = q_online_pred[T.arange(actions.shape[0])[:, None], actions]

        # Double Deep Queu Learning with resulted_states
        q_resulted_state_future = self.q_future.forward(resulted_states)
        q_resulted_state_online = self.q_online.forward(resulted_states)

        max_actions = []
        for i, row in enumerate(q_resulted_state_online):
            max_actions.append(self.get_action(row.clone()))
        # for code efficiency purpose, change it to numpy
        max_actions = np.array(max_actions)
        max_actions = T.tensor(max_actions)
        index_of_q_future = q_resulted_state_future[T.arange(max_actions.shape[0])[:, None], max_actions]
        # If infeasible or not want to repeat the state => done: 1 (I don't want it yet)
        q_resulted_state_future[dones] = 0.0


        # Expand rewards to shape (BATCH_SIZE, NUM_ACTIONS)
        # Q(s,a) = (1-alpha)*Q1(s,a) + alpha * ( reward + gamma * max( Q2'(s',a') ) )
        expanded_rewards = rewards.unsqueeze(1).repeat(1, self.NUM_TOP_SERVICES_SELECT)
        target = expanded_rewards + self.GAMMA * index_of_q_future


        loss = self.q_online.criterion(target, q_online_pred)
        self.loss = loss.item()
        loss.backward()
        self.q_online.optimizer.step()


        self.learning_counter += 1
        self.decrement_epsilon()

        return self.loss
    

    def update(self, w_glob):
        self.q_future.load_state_dict(w_glob)

    def save_models(self):
        self.q_online.save_checkpoint()
        self.q_future.save_checkpoint()

    def load_models(self):
        self.q_online.load_checkpoint()
        self.q_future.load_checkpoint()

    def extract_model(self):
        self.q_online.load_checkpoint()

        return self.q_online

