import numpy as np
import cPickle as pickle
from collections import deque

# Hyperparameters for machine learning algorithm
decay_rate = 0.9 # RMSProp decay rate
learning_rate = 1e-3 # gradient descent learning rate
batch_size = 20 # number of episodes for one update

def load_frozen_brain(frozen_brain):
    return pickle.load(open(frozen_brain + '.p', 'rb'))

class brain: # {{{
    # gamma measures how myopic is the brain (0: myopic, 1: long-sighted)
    # sag measures how long backward in time he remember the situation
    def __init__(self, D = 3, H = [6], O = 2, gamma = 0.99, sag = 3, err = 0.0):
        self.D = D
        self.sag = sag
        self.err = err

        self.prv_obs = deque([])
        for i in range(self.sag):
            self.prv_obs.append(np.zeros(self.D))

        self.buf_x = []
        self.buf_grad_o = []
        self.buf_h = {}
        self.buf_reward = []
        self.episode_N = 0
        self.gamma = gamma

        self.HN = len(H) # number of hidden layers
        self.O = O; # number of actions
        self.model = {}

        self.model[0] = np.random.randn(H[0], D * self.sag) / np.sqrt(D * self.sag) # "Xavier" initialization
        self.buf_h[0] = []
        for i in range(1, self.HN):
            self.model[i] = np.random.randn(H[i], H[i-1]) / np.sqrt(H[i-1])
            self.buf_h[i] = []
        self.model[self.HN] = np.random.randn(O, H[-1]) / np.sqrt(H[-1])

        # The average of gradients over one batch
        self.ttl_grad_W = { k : np.zeros_like(v) for k, v in self.model.iteritems() }
        # rmsprop memory
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() }

    def softmax(self, Y):
        """Compute softmax values for a set of values Yi."""
        e_Y = np.exp(Y)
        return e_Y / e_Y.sum()

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = []
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if(r[t] != r[t]): # reaches the end of episode
                running_add = 0
                continue

            running_add = running_add * self.gamma + r[t]
            discounted_r.append(running_add)
        return np.array(discounted_r[::-1])

    # x: features
    def policy_forward(self, x):
        h = {}
        h[0] = np.dot(self.model[0], x)

        for i in range(1, self.HN):
            h[i] = np.dot(self.model[i], h[i-1])
            h[i][h[i] < 0] = 0 # ReLU nonlinearity

        Q = np.dot(self.model[self.HN], h[self.HN - 1])
        p = self.softmax(Q)

        return p, h # return probabilitys, and hidden states

    # x: episodes x feature_N
    # h: dict of episodes x hidden_N
    # grad_o: episodes x output_N
    def policy_backward(self, x, h, grad_o):
        dW = {}
        dW[self.HN] = np.dot(grad_o.T, h[self.HN - 1])

        # grad_h: xxxx_N x episodes
        grad_h = np.dot(self.model[self.HN].T, grad_o.T)
        grad_h[h[self.HN - 1].T <= 0] = 0 # backprop ReLU

        for i in range(self.HN - 2, -1, -1):
            dW[i + 1] = np.dot(grad_h, h[i])

            grad_h = np.dot(self.model[i + 1].T, grad_h)
            grad_h[h[i].T <= 0] = 0 # backprop ReLU

        dW[0] = np.dot(grad_h, x)

        return dW

    def perform_action(self, feature):
        self.prv_obs.append(feature)
        self.prv_obs.popleft()

        feature = np.hstack(self.prv_obs)

        prob, h = self.policy_forward(feature)

        if(np.random.uniform() < self.err):
            A = np.random.choice(self.O, 1) # Make complete random action
        else :
            A = np.random.choice(self.O, 1, p = prob) # Make stochastic action
        Y = np.zeros(self.O); Y[A] = 1.0

        # Record the action detail for updating the policy
        self.buf_x.append(feature)
        for i in range(0, self.HN):
            self.buf_h[i].append(h[i])
        self.buf_grad_o.append(Y - prob)

        return A

    def receive_feedback(self, reward, done): # done means an episode is finished
        self.buf_reward.append(reward)

        if done: # one episode is finished
            self.prv_obs = deque([])
            for i in range(self.sag):
                self.prv_obs.append(np.zeros(self.D))

            self.episode_N = self.episode_N + 1
            self.buf_reward.append(float('nan'))

            # Finished collecting one batch of data
            if (self.episode_N % batch_size == 0):
                # Prepare for Policy Gradient calculation
                x = np.vstack(self.buf_x)
                self.buf_x = []

                h = {}
                for i in range(0, self.HN):
                    h[i] = np.vstack(self.buf_h[i])
                    self.buf_h[i] = []

                grad_o = np.vstack(self.buf_grad_o)
                self.buf_grad_o = []

                reward = np.vstack(self.buf_reward)
                self.buf_reward = []

                # From immediate-reward to long-term-reward
                long_term_rewards = self.discount_rewards(reward)
                long_term_rewards -= np.mean(long_term_rewards)
                if(np.std(long_term_rewards) > 0.0):
                    long_term_rewards /= np.std(long_term_rewards)

                # Modulate the gradients using long_term_rewards
                grad_o = np.dot(np.diagflat(long_term_rewards), grad_o)
                grad_W = self.policy_backward(x, h, grad_o)

                # Summing over gradient in one batch
                for k in self.model:
                    self.ttl_grad_W[k] += grad_W[k]

                for k, v in self.model.iteritems():
                    g = self.ttl_grad_W[k] # gradient

                    # Using RMSProp to update the Policy Network
                    self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g**2
                    self.model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)

                    self.ttl_grad_W[k] = np.zeros_like(v) # set total gradient stored to 0

                # Pickled the brain
                pickle.dump(self, open('brain_#' + str(id(self)) + '.p', 'wb'))
#}}}

