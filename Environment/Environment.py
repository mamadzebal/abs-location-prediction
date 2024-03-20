import numpy as np


class Environment:
    def __init__(self, NUM_TIME_FRAMES, NUM_SERVICES, NUM_TRAIN_TIME_FRAMES,REWARD_BASE):
        self.NUM_TIME_FRAMES = NUM_TIME_FRAMES
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES
        self.states = []
        self.REWARD_BASE = REWARD_BASE

    def initialize_states(self, initialization_type = 'fixed'):
        # fixed existing services
        fixed_array = np.zeros((self.NUM_SERVICES,))
        if(initialization_type == 'fixed'):
            # fixed_array[0] = 1
            fixed_array[2] = 1
            fixed_array[4] = 1
            fixed_array[5] = 1
            # fixed_array[7] = 1
            # fixed_array[10] = 1
            # fixed_array[13] = 1
            # fixed_array[17] = 1
        time_frames = np.tile(fixed_array, (self.NUM_TIME_FRAMES + 1, 1))


        # randomly assign 0,1 for each services at each time frame
        if(initialization_type == 'random'):
            time_frames = np.random.randint(2, size=(self.NUM_TIME_FRAMES + 1, self.NUM_SERVICES))

            
        elif(initialization_type == 'bernouli'):
            # Set while true in order to not having all 0 for services
            while True:
                # Define the probabilities of each service at each time frame
                probs = np.random.dirichlet(np.ones(self.NUM_SERVICES), size=self.NUM_TIME_FRAMES)
                # Generate the data for each time frame using a Bernoulli distribution
                time_frames = np.zeros((self.NUM_TIME_FRAMES + 1, self.NUM_SERVICES), dtype=np.int)
                for t in range(self.NUM_TIME_FRAMES):
                    time_frames[t+1] = np.random.binomial(1, probs[t])
                if np.sum(time_frames[1]) > 4:
                    break

        # generate a pattern for requests
        elif(initialization_type == 'pattern'):
            PATTERN_LENGTH = 20
            # generate random 0/1 pattern
            # pattern_matrix = np.random.randint(2, size=(PATTERN_LENGTH, self.NUM_SERVICES))

            # create pattern matrix
            pattern_matrix = np.zeros((PATTERN_LENGTH, self.NUM_SERVICES))
            # num_ones = int(PATTERN_LENGTH * 0.6)  # set desired number of ones in each column
            # for j in range(self.NUM_SERVICES):
            #     pattern_matrix[:num_ones,j] = 1
            #     np.random.shuffle(pattern_matrix[:,j])
            
            # pattern_matrix[0] = np.array([0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,1])
            # pattern_matrix[1] = np.array([0,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1])
            # pattern_matrix[2] = np.array([0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,0,0])
            # pattern_matrix[3] = np.array([1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0])
            # pattern_matrix[4] = np.array([1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0])
            # pattern_matrix[5] = np.array([0,1,0,0,0,0,1,1,0,1,0,1,0,0,1,1,0,0,0,1])
            # pattern_matrix[6] = np.array([0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1])
            # pattern_matrix[7] = np.array([1,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1])
            # pattern_matrix[8] = np.array([0,1,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0])
            # pattern_matrix[9] = np.array([0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,0,0,1])

            pattern_matrix[0]   = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[1]   = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[2]   = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[3]   = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[4]   = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[5]   = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[6]   = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[7]   = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[8]   = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[9]   = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
            pattern_matrix[10]  = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
            pattern_matrix[11]  = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
            pattern_matrix[12]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
            pattern_matrix[13]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
            pattern_matrix[14]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
            pattern_matrix[15]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
            pattern_matrix[16]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
            pattern_matrix[17]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
            pattern_matrix[18]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
            pattern_matrix[19]  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

            # repeat pattern to cover all rows
            num_repeats = int(np.ceil((self.NUM_TIME_FRAMES+1)/PATTERN_LENGTH))
            repeated_pattern = np.tile(pattern_matrix, (num_repeats, 1))

            # trim to correct number of rows
            time_frames = repeated_pattern[:self.NUM_TIME_FRAMES+1]
        

        return time_frames
    
    def get_state(self, states, next_time):
        return states[next_time-self.NUM_TRAIN_TIME_FRAMES:next_time]
    
    def step(self, states, actions, next_time = 0):
        reward = 0

        # We postulate that even services (2,4,6,...) are TSN ones with higher priority! 
        # for each correct service index => reward += 1/0.5 and for each incorrect service index => reward -= 1/0.3
        for i in range(len(actions)):
            action_index = actions[i]
            if states[next_time][action_index] == 1:
                # reward += self.REWARD_BASE if(action_index % 2 == 0) else self.REWARD_BASE/2
                reward += self.REWARD_BASE
            else:
                # reward -= self.REWARD_BASE if(action_index % 2 == 0) else self.REWARD_BASE/2
                reward -= self.REWARD_BASE

        resulted_state = self.get_state(states, next_time)

        return resulted_state, reward