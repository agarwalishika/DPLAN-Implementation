from rl.core import Processor
from rl.util import clone_model
from rl.memory import Memory, SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.callbacks import Callback
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from ADEnv import ADEnv

class QNetwork(Model):
    """
    Network architecture with one hidden layer.
    """
    def __init__(self,input_shape, hidden_unit=20):
        super().__init__()
        self.input=Input(shape=input_shape)
        self.hidden=Dense(hidden_unit,activation="relu",kernel_regularizer=regularizers.l2(0.01))
        self.output=Dense(2,activation="linear")

    def penulti_layer(self,state):
        x=self.input(state)
        x=self.hidden(x)

        # output of penultilayer
        return x

    def call(self,state):
        x=self.input(state)
        x=self.hidden(x)
        x=self.output(x)

        return x

def DQN_iforest(x, DQN: QNetwork):
    # iforest function on the penuli-layer space of DQN
    latent_x=DQN.penulti_layer(x).detach()
    # calculate anomaly scores in the latent space
    iforest=IsolationForest().fit(latent_x)
    scores=iforest.score_samples(latent_x)
    # scaler scores to [0,1]
    scaler=MinMaxScaler()
    norm_scaler=scaler.fit_transform(scores)

    return norm_scaler


class DPLANProcessor(Processor):
    """
    Customize the fit function of DQNAgent.
    """
    def __init__(self, env: ADEnv):
        """
        :param env: Used to get the dataset from the environment.
        """
        self.x=env.x
        self.intrinsic_reward=None

        # store the index of s_t
        self.last_observation=None

    def process_step(self, observation, reward, done, info):
        # note that process_step runs after the step of environment
        # if we only modify the process_observation function,
        # the last_observation attribute will change to s_t+1 before the intrinsic reward is added.
        last_observation=self.last_observation # stored beore changed.

        observation=self.process_observation(observation)
        reward=self.process_reward(reward,last_observation)
        info=self.process_info(info)

        return observation, reward, done, info

    def process_observation(self,observation):
        # note that the observation generated from ADEnv is the index of the point in dataset
        # convert it to a numpy array
        self.last_observation=observation

        return self.x[observation,:]

    def process_reward(self, reward_e, last_observation):
        # integrate the intrinsic reward function
        reward_i=self.intrinsic_reward[last_observation]

        return reward_e+reward_i

class DPLANCallbacks(Callback):
    def on_train_begin(self, logs=None):
        # calculate the intrinsic_reward from the initialized DQN
        self.model.processor.intrinsic_reward=DQN_iforest(self.env.x, self.model)

    def on_episode_end(self, episode, logs={}):
        # on the end of episode, DPLAN needs to update the target DQN and the penulti-features
        # the update process of target DQN have implemented in "rl.agents.dqn.DQNAgent.backward()"
        self.model.processor.intrinsic_reward=DQN_iforest(self.env.x, self.model)


def DPLAN(env: ADEnv, settings: dict, testdata, *args, **kwargs):
    """
    1. Train a DPLAN model on anomaly-detection environment.
    2. Test it on the test dataset.
    3. Return the predictions.
    :param env: Environment of the anomaly detection.
    :param settings: Settings of hyperparameters in dict format.
    :param testdata: Test dataset.
    :param n_run:
    """
    # hyperparameters
    l=settings["hidden_layer"]
    M=settings["memory_size"]
    warmup_steps=settings["warmup_steps"]
    n_episodes=settings["episodes"]
    n_steps_episode=settings["steps_per_episode"]
    max_epsilon=settings["epsilon_max"]
    min_epsilon=settings["epsilon_min"]
    greedy_course=settings["epsilon_course"]
    minibatch_size=settings["minibatch_size"]
    gamma=settings["discount_factor"]
    lr=settings["learning_rate"]
    min_grad=settings["minsquared_gradient"]
    grad_momentum=settings["gradient_momentum"]
    N=settings["penulti_update"] # hyper-parameter not used
    K=settings["target_update"]

    # initialize DQN Agent
    input_shape=env.n
    n_actions=env.action_space.n
    model=QNetwork(input_shape=input_shape,
                   hidden_unit=l)
    policy=LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                attr='eps',
                                value_max=max_epsilon,
                                value_min=min_epsilon,
                                value_test=0.,
                                nb_steps=greedy_course)
    memory=SequentialMemory(limit=M,
                            window_length=1)
    processor=DPLANProcessor()
    agent=DQNAgent(model=model,
                   policy=policy,
                   nb_actions=n_actions,
                   memory=memory,
                   processor=processor,
                   gamma=gamma,
                   batch_size=minibatch_size,
                   nb_steps_warmup=warmup_steps,
                   target_model_update=K,
                   )
    optimizer=RMSprop(learning_rate=lr, clipnorm=1.,momentum=grad_momentum)
    agent.compile(optimizer=optimizer)
    # initialize target DQN with weight=0


    # train DPLAN
    callbacks=DPLANCallbacks()
    agent.fit(env=env,
              nb_steps=warmup_steps+n_episodes*n_steps_episode,
              callbacks=callbacks,
              nb_max_episode_steps=n_steps_episode)