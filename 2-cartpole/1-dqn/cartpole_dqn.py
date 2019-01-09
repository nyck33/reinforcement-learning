import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        #Tried to change the way epsilon is calculated per Q-learning-cart.ipynb but doesn't work here
        #although epsilon has a gradual decline for this game we want a fast transition from explore to exploit
        #self.epsilon_start = 1.0
        self.epsilon = 1 #0
        self.epsilon_decay = 0.999 #0.0001
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform')) #std distribution from [-limit, limit], limit=sqrt(6/num input units in weight tensor)
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', #TODO: defind activation
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) #TODO: deine loss function mse
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights()) #using same weights as model, so until memory is full weights don't change

    # get action from model using epsilon-greedy policy
    def get_action(self, state, step):
        #self.epsilon = (self.epsilon_min + (self.epsilon_start \
         #       - self.epsilon_min) *np.exp(-self.epsilon_decay*step))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else: #exploit more later
            q_value = self.model.predict(state)
            return np.argmax(q_value[0]) 

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done, step):
        self.memory.append((state, action, reward, next_state, done))
        #if epsilon has not reached min and we are done filling memory
        if (self.epsilon > self.epsilon_min) and (len(self.memory) >= self.train_start):
            self.epsilon *= self.epsilon_decay #epsilon decays during the memory filling
            #self.epsilon = self.epsilon_min + (self.epsilon_start \
              #  - self.epsilon_min) *np.exp(-self.epsilon_decay*step)
    # pick samples randomly from replay memory (with batch_size)
    #epsilon decays while the memory buffer is being filled
    def train_model(self):
        if len(self.memory) < self.train_start: #memory starts at 2000, train_start at 1000
            return
        if (len(self.memory)==(self.train_start)):
            print(self.epsilon)
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        #initialize these arrays
        update_input = np.zeros((batch_size, self.state_size)) #np array for the state value
        update_target = np.zeros((batch_size, self.state_size)) #np array for target state value
        action, reward, done = [], [], [] #create empty lists

        for i in range(self.batch_size): #updating the parameters used from batch
            update_input[i] = mini_batch[i][0] #state
            action.append(mini_batch[i][1]) #action
            reward.append(mini_batch[i][2]) 
            update_target[i] = mini_batch[i][3] #next_state
            done.append(mini_batch[i][4]) #done

        #updating model and target model    
        target = self.model.predict(update_input) #.predict generates predictions for input
        target_val = self.target_model.predict(update_target) #this exists only to find the greedy action???

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i] #Q value is the reward if done
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * ( #else take greedy action
                    np.amax(target_val[i]))

        # and do the model fit train data x on labels y for batch_size (trains model/gradient descent using Adam)
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n


    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size]) #reshape state into array of one row and state_size num cols
        steps = 0 #up to 500

        while not done:
            if agent.render: #if True
                env.render()
            steps+=1
            # get action for the current state and go one step in environment
            action = agent.get_action(state, steps)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100 because it's a continuous task
            reward = reward if not done or score == 499 else -100 #if done, -100. 499 ensures -100 doesn't occur after completion of 500
            #-100 is a big punishment for falling since rewards are small and cumulative through 500 steps
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done, steps)
            # every time step do the training, won't start real until deque is at least traing_start or 1000 full
            agent.train_model()
            score += reward
            state = next_state #updated state

            if done:
                # every episode update the target model to be same with model (donkey and carrot), carries over to next episdoe
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100 #if done, give back the 100 that was taken away when appending rewards above
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b') #'b' is type of marking for plot
                pylab.savefig("./save_graph/cartpole_dqn.png")
                if(e % 10 == 0):
                    print("episode:", e, "  score:", score, "  memory length:",
                        len(agent.memory), "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model every 50th episode
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_dqn.h5")
