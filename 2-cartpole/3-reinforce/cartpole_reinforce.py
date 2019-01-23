import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


EPISODES = 1#1000


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99 #discount rewards
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        #first layer is a state_size of 4 so 4 neurons?
        #glorot_uniform is Xavier uniform initializer draws samples from uniform distrib, etc.
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        #24 neurons
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        #24 neurons
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        #output of 2 = right or left action
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten() #get probabilities
        print("get_action policy", policy)
        return np.random.choice(self.action_size, 1, p=policy)[0] 

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards) #return array of zeros same shape and type 
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        print("discounted_rewards", discounted_rewards)
        return discounted_rewards

    # save <s, a ,r> of each step in their own lists
    def append_sample(self, state, action, reward):
        self.states.append(state)
        print("ap self.states", self.states)
        self.rewards.append(reward)
        print("ap self.rewards", self.rewards)
        self.actions.append(action)
        print("ap self.actions", self.actions)
    # update policy network every episode at the end
    def train_model(self):
        episode_length = len(self.states) #list 1D
        print("episode_length", episode_length)
        #discount the rewards array and reversed
        discounted_rewards = self.discount_rewards(self.rewards)
        #minus mean so here disc_rewards is somewhat = advantage
        discounted_rewards -= np.mean(discounted_rewards) #minus the mean
        #std = sqrt(mean(abs(x - x.mean())**2)).  rewards are multiple of std. dev.
        discounted_rewards /= np.std(discounted_rewards)
        print("std dev disc_rewards", discounted_rewards)
        update_inputs = np.zeros((episode_length, self.state_size)) #no dtype= so this is 2D
        advantages = np.zeros((episode_length, self.action_size)) #2D
        #print("advantages shape", np.shape(advantages)) 
        #print("advantages", advantages)

        for i in range(episode_length):
            #copying states array to update_inputs
            update_inputs[i] = self.states[i] #start from state zero
            #print("update_inputs[{}]".format(i), update_inputs[i])
            #filling in squashed rewards into advantages for each action at each state
            advantages[i][self.actions[i]] = discounted_rewards[i]  
            #print("advantages 2D", advantages[i][self.actions[i]])
            #print("\n\nupdate_inputs final", update_inputs)
            #print("\n\nadvantages final {}".format(advantages))
        #(training data=states, targets=advantages which is like the action-value)
        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        #clear out states, actions, rewards for next episode
        self.states, self.actions, self.rewards = [], [], [] 

if __name__ == "__main__":
    
    print("\n\n\n\n\n\nstart\n\n")
    # In case of CartPole-v1, you can play until 500 time step, says >200 ep length
    env = gym.make('CartPole-v1') #Cartpole-v0 continuous
    # get size of state and action from environment
    state_size = env.observation_space.shape[0] #4 cart position, vel, pole angle and rotation
    print("state_size", state_size)
    action_size = env.action_space.n
    print("action_size", action_size) #2, push right or left
    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        print("state reset", state)
        #why reshape if already (1,4)?
        state = np.reshape(state, [1, state_size]) #reshape into 1,4
        print("np.shape state", np.shape(state)) #(1,4) 1 row, 4 columns
        print("state", state)

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            print("action", action)
            next_state, reward, done, info = env.step(action)
            print("env.step", next_state, reward, done)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100
            print("reward", reward)
            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            print("score", score)
            state = next_state
            print("state=next_state", state)

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                print("done score", score)
                scores.append(score)
                print("done scores", scores)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_reinforce.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_reinforce.h5")
