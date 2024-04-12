from Model.DRL import ABS_DRL
from Output.PlotResult import plot_reward
from Output.PlotLoss import plot_loss
import os

# Define the main parameters (state, action, and reward variables)
NUM_TIME_FRAMES = 5000
NUM_TRAIN_TIME_FRAMES = 100
CANDIDATE_LOCATIONS = 25

# Agent parameters
EPSILON = 1
EPSILON_MIN = 0.0001
EPSILON_DEC = 5e-4
GAMMA = 0.75
LR = 0.001
WEIGHT_DECAY_LAMBDA = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 1000000
REPLACE_COUNTER = 1000
NUM_TOP_SERVICES_SELECT = 1
CHECKPOINT_DIR = 'tmp-learning-models/'

# Environment parameters
REWARD_BASE = 1

# Model parameters
H_LAYER1_DIMENSION = 128
H_LAYER2_DIMENSION = 64
H_LAYER3_DIMENSION = 128
H_LAYER4_DIMENSION = 512
H_LAYER5_DIMENSION = 512


if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

def DRL_TRAIN(mg_drl, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS):
    total_avg_rewards, bst_rwd, gamma, epsilon_min, lr = mg_drl.drl_alloc_train()
    plot_reward(total_avg_rewards, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, gamma, epsilon_min, lr, 'train')
    plot_loss(mg_drl.batch_loss)
    

def DRL_EVAL(mg_drl, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, PATTERN_STARTING_POINT):
    actions = mg_drl.drl_alloc_eval(starting_point=PATTERN_STARTING_POINT)
    print(actions)
    # total_avg_rewards, bst_rwd, gamma, epsilon_min, lr = mg_drl.drl_alloc_eval()
    # plot_reward(total_avg_rewards, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, gamma, epsilon_min, lr, 'eval')


print("****************************** START *********************************")

# We are doing RNN(Recurrent Neural Network) which learn time series
NUM_ABS = 1
for iteration in range(NUM_ABS):
    # Initiate Deep Reinforcement Learning
    mg_drl = ABS_DRL(
        NUM_TIME_FRAMES=NUM_TIME_FRAMES, CANDIDATE_LOCATIONS=CANDIDATE_LOCATIONS, NUM_TRAIN_TIME_FRAMES=NUM_TRAIN_TIME_FRAMES, ITERATION=iteration,
        EPSILON=EPSILON, EPSILON_MIN=EPSILON_MIN, EPSILON_DEC=EPSILON_DEC, GAMMA=GAMMA, LR=LR, WEIGHT_DECAY_LAMBDA=WEIGHT_DECAY_LAMBDA, BATCH_SIZE=BATCH_SIZE, MEMORY_SIZE=MEMORY_SIZE, REPLACE_COUNTER=REPLACE_COUNTER,
        NUM_TOP_SERVICES_SELECT=NUM_TOP_SERVICES_SELECT, CHECKPOINT_DIR=CHECKPOINT_DIR,
        REWARD_BASE=REWARD_BASE,
        H_LAYER1_DIMENSION=H_LAYER1_DIMENSION, H_LAYER2_DIMENSION=H_LAYER2_DIMENSION, H_LAYER3_DIMENSION=H_LAYER3_DIMENSION, H_LAYER4_DIMENSION=H_LAYER4_DIMENSION, H_LAYER5_DIMENSION=H_LAYER5_DIMENSION
    )
    DRL_TRAIN(mg_drl, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS)

# evaluate
# PATTERN_STARTING_POINT = 0
# DRL_EVAL(mg_drl, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, PATTERN_STARTING_POINT)

print("****************************** FINISH *********************************")
