from Environment.Agent import Agent
from Environment.Environment import Environment
from Federated.HFL import HFL
import numpy as np
import random
import sys
import time

class ABS_DRL(object):
    def __init__(
            self, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, NUM_TRAIN_TIME_FRAMES, ITERATION,
            EPSILON, EPSILON_MIN, EPSILON_DEC, GAMMA, LR, WEIGHT_DECAY_LAMBDA, BATCH_SIZE, MEMORY_SIZE, REPLACE_COUNTER,
            NUM_TOP_SERVICES_SELECT, CHECKPOINT_DIR,
            REWARD_BASE,
            H_LAYER1_DIMENSION, H_LAYER2_DIMENSION, H_LAYER3_DIMENSION, H_LAYER4_DIMENSION, H_LAYER5_DIMENSION
        ):

        # Agent parameters
        self.BATCH_SIZE = BATCH_SIZE
        self.ITERATION = ITERATION
        

        # DRL parameters
        self.NUM_TIME_FRAMES = NUM_TIME_FRAMES
        self.CANDIDATE_LOCATIONS = CANDIDATE_LOCATIONS
        self.NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES
        self.FILE_NAME = "agnt"

        self.env_obj = Environment(NUM_TIME_FRAMES = self.NUM_TIME_FRAMES, CANDIDATE_LOCATIONS = CANDIDATE_LOCATIONS, NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES, REWARD_BASE=REWARD_BASE, CHECKPOINT_DIR=CHECKPOINT_DIR)
        self.agent = Agent(
            NUM_ACTIONS = self.CANDIDATE_LOCATIONS, INPUT_SHAPE = self.NUM_TRAIN_TIME_FRAMES * self.CANDIDATE_LOCATIONS, NUM_TRAIN_TIME_FRAMES = self.NUM_TRAIN_TIME_FRAMES, NAME = self.FILE_NAME, ITERATION = self.ITERATION,
            EPSILON=EPSILON, EPSILON_MIN=EPSILON_MIN, EPSILON_DEC=EPSILON_DEC, GAMMA=GAMMA, LR=LR, WEIGHT_DECAY_LAMBDA=WEIGHT_DECAY_LAMBDA, BATCH_SIZE=self.BATCH_SIZE, MEMORY_SIZE=MEMORY_SIZE, REPLACE_COUNTER=REPLACE_COUNTER,
            NUM_TOP_SERVICES_SELECT=NUM_TOP_SERVICES_SELECT, CHECKPOINT_DIR=CHECKPOINT_DIR,
            H_LAYER1_DIMENSION=H_LAYER1_DIMENSION, H_LAYER2_DIMENSION=H_LAYER2_DIMENSION, H_LAYER3_DIMENSION=H_LAYER3_DIMENSION, H_LAYER4_DIMENSION=H_LAYER4_DIMENSION, H_LAYER5_DIMENSION=H_LAYER5_DIMENSION
        )
        
        self.batch_loss = []
        
        # TODO: get parameter from global side
        self.NUM_FEDERATED_USERS = 2

        # TODO: remove parameters from agent and set in this place (all of them)
        self.federated = HFL(
            NUM_USERS=self.NUM_FEDERATED_USERS, LR=0.001, WEIGHT_DECAY_LAMBDA=0.0001, NUM_ACTIONS=self.CANDIDATE_LOCATIONS, INPUT_SHAPE = self.NUM_TRAIN_TIME_FRAMES * self.CANDIDATE_LOCATIONS, FILE_NAME=self.FILE_NAME + "_q_hfl", CHECKPOINT_DIR='tmp-learning-models/', NUM_TRAIN_TIME_FRAMES = self.NUM_TRAIN_TIME_FRAMES, BATCH_SIZE=self.BATCH_SIZE
        )

    def drl_each_step(self, learn, NUM_TIME_FRAMES, NUM_TRAIN_TIME_FRAMES, states, agent, w_old):
        total_avg_rewards = []
        bst_rwd = -np.inf
        
        game_stps = 0
        rwds, epss, stps = [], [], []
        game_rwd = 0
        check_each_j = 100


        update_w = {}
        for j in range(NUM_TRAIN_TIME_FRAMES, len(states)-1):
            start_t = time.time()

            state = (states[j-NUM_TRAIN_TIME_FRAMES:j])            

            ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))
            actions = agent.get_top_predicted_services(state, ACTION_SEED, learn)
            resulted_state, the_action_rwd = self.env_obj.step(states, actions, (j+1))

            if(learn):
                done = False
                agent.store_transition(state, actions, the_action_rwd, resulted_state, done)
                each_batch_loss_j = agent.learn()

                if(each_batch_loss_j != 'empty'):
                    print(
                        'episode: ', j, '|',
                        'learning counter: %d' % agent.learning_counter, '|',
                        'eps: %.4f' % agent.EPSILON, '|',
                        'loss: %.4f' % agent.loss, '|',
                        'steps: %d' % game_stps, '|',
                        'learning time: %.2f seconds' % (time.time() - start_t), '|',
                        'selected action: ', actions, '|',
                        'reward: ', the_action_rwd
                    )

                    self.batch_loss.append(each_batch_loss_j)

                    w_new = agent.q_future.state_dict()
                    for k in w_new.keys():
                        update_w[k] = w_new[k] - w_old[k]

            game_rwd += the_action_rwd
            game_stps += 1
            rwds.append(the_action_rwd)
            
            if(j % check_each_j == 0):
                stps.append(game_stps)
                epss.append(agent.EPSILON)
                avg_rwd = np.mean(rwds[-check_each_j:])
                # print(
                #     'episode:', j,
                #     'reward: %.2f' % avg_rwd,
                #     'eps: %.4f' % agent.EPSILON,
                #     'steps:', game_stps
                # )
                if avg_rwd > bst_rwd:
                    bst_rwd = avg_rwd
                    if(learn):
                        agent.save_models()
                    
                

                total_avg_rewards.append(avg_rwd)

        return total_avg_rewards, bst_rwd, update_w, 1 if(len(self.batch_loss) == 0) else (sum(self.batch_loss) / len(self.batch_loss))


    def run_federated_learning(self):
        self.env_obj.states = self.env_obj.initialize_states('fixed')
        clients, server, each_client_idxs = self.federated.create_client_server(self.env_obj.states)


        # TODO: run each client in different thread


        NUM_ITERATION_EPOCHS = 1
        for iter in range(NUM_ITERATION_EPOCHS):
            epoch_start = time.time()

            server.clients_update_w, server.clients_loss = [], []
            for idx in range(self.NUM_FEDERATED_USERS):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start for the ", idx ," th client")
                total_avg_rewards, bst_rwd, update_w, loss = self.drl_each_step(
                    learn=True, 
                    NUM_TIME_FRAMES=self.NUM_TIME_FRAMES, 
                    NUM_TRAIN_TIME_FRAMES=self.NUM_TRAIN_TIME_FRAMES, 
                    states=self.env_obj.states[list(each_client_idxs[idx])],
                    agent=clients[idx], 
                    w_old=clients[idx].q_future.state_dict()
                )
                server.clients_update_w.append(update_w)
                server.clients_loss.append(loss)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end for the ", idx ," th client with loss: ", loss)
            
            # calculate global weights
            w_glob, loss_glob = server.FedAvg()

            # update local weights
            for idx in range(self.NUM_FEDERATED_USERS):
                clients[idx].update(w_glob)

            epoch_end = time.time()

            print('Training time:', epoch_end - epoch_start)


    def drl_alloc_train(self):
        self.env_obj.states, pattern_matrix = self.env_obj.initialize_states('pattern', learn=True)
        self.env_obj.save_state(pattern_matrix, 'abs_' + str(self.ITERATION))
        total_avg_rewards, bst_rwd, update_w, loss = self.drl_each_step(
            learn=True, NUM_TIME_FRAMES=self.NUM_TIME_FRAMES, NUM_TRAIN_TIME_FRAMES=self.NUM_TRAIN_TIME_FRAMES, states=self.env_obj.states, agent=self.agent, w_old=self.agent.q_future.state_dict()
        )
            
        return total_avg_rewards, bst_rwd, self.agent.GAMMA, self.agent.EPSILON_MIN, self.agent.lr

    def drl_alloc_eval(self, starting_point):
        self.env_obj.states = self.env_obj.initialize_states('pattern', learn=False, starting_point=starting_point)
        self.agent.load_models()
        self.agent.EPSILON = 0
        self.agent.EPSILON_MIN = 0
        

        ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))
        actions = self.agent.get_top_predicted_services(self.env_obj.states, ACTION_SEED, False)

        # total_avg_rewards, bst_rwd, update_w, loss = self.drl_each_step(learn=False, NUM_TIME_FRAMES=self.NUM_TIME_FRAMES, NUM_TRAIN_TIME_FRAMES=self.NUM_TRAIN_TIME_FRAMES, states=self.env_obj.states, agent=self.agent, w_old=self.agent.q_future.state_dict())
        
        return actions
