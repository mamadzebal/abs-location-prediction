import numpy as np
import os

class Environment:
    def __init__(self, NUM_TIME_FRAMES, CANDIDATE_LOCATIONS, NUM_TRAIN_TIME_FRAMES,REWARD_BASE, CHECKPOINT_DIR):
        self.NUM_TIME_FRAMES = NUM_TIME_FRAMES
        self.CANDIDATE_LOCATIONS = CANDIDATE_LOCATIONS
        self.NUM_TRAIN_TIME_FRAMES = NUM_TRAIN_TIME_FRAMES
        self.states = []
        self.REWARD_BASE = REWARD_BASE
        self.checkpoint_dir = CHECKPOINT_DIR

    def generate_random_location_vector(self):
        # Initialize a vector with zeros
        location_vector = np.zeros(self.CANDIDATE_LOCATIONS)
        
        # Set one random location to 1
        location_vector[np.random.randint(self.CANDIDATE_LOCATIONS)] = 1
        
        return location_vector

    def generate_biased_location_vector(self):
        # Initialize a vector with zeros
        location_vector = np.zeros(self.CANDIDATE_LOCATIONS)
        
        # TODO: update the probability based on CANDIDATE LOCATIONS
        # Fill outer squares with high probabilities
        for i in range(5):
            high_prob_indices = [i, 20+i, i*5, i*5+4]
            location_vector[np.random.choice(high_prob_indices)] = 1
        
        # Fill second outer square with high probabilities
        for j in [6, 7, 8, 11, 13, 16, 17, 18]:
            location_vector[j] = np.random.choice([0, 1], p=[0.5, 0.5])
        
        # Ensure only one location is 1
        num_ones = np.sum(location_vector)
        if num_ones > 1:
            indices = np.where(location_vector == 1)[0]
            chosen_index = np.random.choice(indices)
            location_vector[:] = 0
            location_vector[chosen_index] = 1
        
        return location_vector

    def save_state(self, state, filename):
        checkpoint_file = os.path.join(self.checkpoint_dir, filename)
        np.save(checkpoint_file, state)

    def load_state(self, filename):
        checkpoint_file = os.path.join(self.checkpoint_dir, filename)
        pattern_matrix = np.load(checkpoint_file)
        
        return pattern_matrix

    def initialize_states(self, initialization_type = 'fixed', learn = True, starting_point = 0):
        num_pattern = self.NUM_TIME_FRAMES if(learn) else self.NUM_TRAIN_TIME_FRAMES
        
        # generate a pattern for requests
        if(initialization_type == 'pattern'):
            PATTERN_LENGTH = np.random.randint(5, 11)
        
            # create pattern matrix
            pattern_matrix = np.zeros((PATTERN_LENGTH, self.CANDIDATE_LOCATIONS))

            for j in range(PATTERN_LENGTH):
                pattern_matrix[j] = self.generate_biased_location_vector()

            # pattern_matrix[0]   = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # pattern_matrix[1]   = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
            # pattern_matrix[2]   = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
            # pattern_matrix[3]   = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
            # pattern_matrix[4]   = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # pattern_matrix[5]   = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            
            # Reorder pattern matrix based on the starting point
            pattern_matrix = np.roll(pattern_matrix, -starting_point, axis=0)

            # repeat pattern to cover all rows
            num_repeats = int(np.ceil((num_pattern+1)/PATTERN_LENGTH))
            repeated_pattern = np.tile(pattern_matrix, (num_repeats, 1))


            # trim to correct number of rows
            time_frames = repeated_pattern[:num_pattern+1]
        

        return time_frames, pattern_matrix
    
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