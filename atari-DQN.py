import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.cuda import comm
from torchvision import transforms
import numpy as np
import os

DEVICE = torch.device('cuda')
DEVICE_COUNT = torch.cuda.device_count()

LOAD_FROM_FILE = False
model_path = "model/"

game = "pong"

games = {"beam_rider":"roms/Beamrider.bin",
         "breakout": "roms/Breakout - Breakaway IV.bin",
         "enduro":"roms/Enduro.bin",
         "pong":"roms/Video Olympics - Pong Sports.bin",
         "qbert":"roms/Q-bert.bin",
         "seaquest":"roms/Seaquest.bin",
         "space_invaders":"roms/Space Invaders.bin"         
        }

class QModel(nn.Module):
    def __init__(self, num_images, image_size, num_actions):
        """Constructs QModel.

        Assumes input dimension of (N, num_images, image_size[0], image_size[1]) where
            N is the size of the minibatch
            num_images is the number of images in each state
            image_size is the size of each image in a state
        In the original paper (image_size, image_size, num_images) format is
        being used.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(num_images, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # calculate the correct dimensions for fully connected layer
        
        tmp = torch.zeros((1, num_images) +  image_size)
        tmp = torch.flatten(self.conv2(self.conv1(tmp)))
        input_dim = tmp.shape[0]
        print("creating QModel with fully connected layer 1 input dim: ", input_dim)
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.out(x)
        
class ReplayStorage:
    def __init__(self, capacity, state_size, num_actions, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_added = 0
        # self.states[i][0:4] is the start state
        # and self.states[i][1:] is the end state
        #self.states = torch.zeros((capacity,) + (state_size[0] + 1, state_size[1], state_size[2]), dtype=torch.uint8, device=DEVICE)
        self.states = comm.scatter(torch.zeros((capacity,) + (state_size[0] + 1, state_size[1], state_size[2]), dtype=torch.uint8),devices=range(DEVICE_COUNT))
        self.actions = torch.zeros((capacity,) + (num_actions, ), device=DEVICE)
        self.rewards = torch.zeros((capacity, ), device=DEVICE)
        self.is_terminals = torch.zeros((capacity, ), dtype=torch.bool)
        
    def sample_minibatch(self, random_gen):
        """Returns a random minibatch from the storage.
        
        Parameters:
            random_gen: np.random.Generator
        """
        assert self.n_added > 0
        limit = min(self.n_added, self.capacity)
        selection = torch.randint(0, limit, (self.batch_size, ))
        #gather selection states to one tensor
        selection_device = torch.div(selection*DEVICE_COUNT, self.capacity, rounding_mode='floor')
        selected_states = comm.gather([
            self.states[device_index][selection[selection_device==device_index] \
                                 - device_index * (self.capacity // DEVICE_COUNT) \
                                 - min(device_index, self.capacity % DEVICE_COUNT)]
            for device_index in range(DEVICE_COUNT)])
        #start_sample = self.states[selection, :4, ...].to(DEVICE)
        start_sample = selected_states[:,:4,...]
        action_sample = self.actions[selection]
        reward_sample = self.rewards[selection]
        #end_sample = self.states[selection, 1:, ...].to(DEVICE)
        end_sample = selected_states[:,1:,...]
        terminal_sample = self.is_terminals[selection]
        return start_sample.float()/255, action_sample, reward_sample, end_sample.float()/255, terminal_sample

    def add_transition(self, start_state, action, reward, end_state):
        """Adds a transition to the storage."""

        assert torch.abs(torch.sum(start_state.tensor[1:] - end_state.tensor[:3])) < 1e-5
        pos = self.n_added % self.capacity
        device_index = (pos*DEVICE_COUNT) // self.capacity
        device_pos = pos - device_index * (self.capacity // DEVICE_COUNT) \
                         - min(device_index, self.capacity % DEVICE_COUNT)
        #self.states[pos, :4] = 255 * start_state.tensor
        self.states[device_index][device_pos, :4] = 255 * start_state.tensor
        self.actions[pos] = action
        self.rewards[pos] = reward
        #self.states[pos, 4] = 255 * end_state.tensor[3]
        self.states[device_index][device_pos, 4] = 255 * end_state.tensor[3]
        self.is_terminals[pos] = end_state.is_terminal
        self.n_added += 1



# Maybe a bit useless class because we anyway need to deal directely with the tensors
class State:
    def __init__(self, num_images, image_size):
        self.tensor = torch.zeros((num_images,) + image_size, device=DEVICE)
        self.is_terminal = False
    
    def update(self, image, is_terminal):

        self.tensor = torch.roll(self.tensor, -1, 0)
        self.tensor[-1] = image
        self.is_terminal = is_terminal
        
    def copy(self):
        tmp = State(self.tensor.shape[0], self.tensor.shape[1:])
        tmp.tensor = self.tensor.detach().clone()
        tmp.is_terminal = self.is_terminal
        return tmp

class LinearScheduler:
    """Linear scheduler which stays constant after reaching end of the linear area"""
    def __init__(self, start_value, end_value, length):
        self.start_value = start_value
        self.end_value = end_value
        self.length = length
        self.pos = 0
    
    def tick(self):
        self.pos += 1

    def value(self):
        if self.pos >= self.length:
            return self.end_value
        
        return self.start_value + (self.end_value - self.start_value) * self.pos / self.length
    


class Agent:
    def __init__(self, q_model, optimizer, init_state, storage, num_actions, eps_scheduler, gamma):
        """Constructs Agent.

        Parameters:
            q_model: a neural network returning values of the Q function
            optimizer: a pytorch Optimizer for q_model
            init_state: State object
                The agent state at the beginning of an episode
            storage: ReplayStorage
            num_actions: an integer
                The number of actions available for the agent
            eps_scheduler: Scheduler
                eps.value() should be the probability of taking a random move instead of the current
                best one.
                eps.tick() called at every inform_response call
            gamma: a scalar
                The reward discount rate
        """

        self.q_model = q_model
        self.optimizer = optimizer
        self.state = init_state.copy()
        self._init_state = init_state.copy()
        self.storage = storage
        self.num_actions = num_actions
        self.eps_scheduler = eps_scheduler
        self.gamma = gamma

    def reset(self):
        self.state = self._init_state.copy() 
    
    def get_action(self, random_gen, epsilon='auto'):
        """Returns action taken by the agent at its current state.
        
        Parameters:
            random_gen: np.random.Generator
            epsilon: 'auto' or a scalar
                if 'auto' then the self.eps_scheduler is used
        """
        if epsilon == 'auto':
            epsilon = self.eps_scheduler.value()

        if random_gen.random() < epsilon:
            return random_gen.integers(self.num_actions)
        
        # add extra dimension to the tensor

        tmp = torch.unsqueeze(self.state.tensor, 0)
        return torch.argmax(torch.flatten(self.q_model(tmp))).item()
    
    def inform_response(self, action_idx, reward, image, is_terminal, save_transition=True):
        """Informs the agent of the consequences of `action`.

        Updates agent state and optinally adds the transition to the ReplayStorage

        Parameters:
            action_idx: an integer
            reward: a scalar
                The immediate reward
            image: torch.Tensor
                The preprocessed image observed by the agent after `action`
            is_terminal: boolean
                True if the new state is terminal
            save_transition: boolean
        """

        old_state = self.state.copy()
        self.state.update(image, is_terminal)
        if save_transition:
            # one hot encoded action
            action = torch.zeros((self.num_actions, ), device=DEVICE)
            action[action_idx] = 1.0

            self.storage.add_transition(old_state, action, reward, self.state.copy())
    
    def train_step(self, random_gen):
        """Applies one gradient descent step"""
        
        self.optimizer.zero_grad()

        start_states, actions, rewards, end_states, is_terminals = self.storage.sample_minibatch(random_gen)
        ys = rewards.clone()
        #print(is_terminals)
        ys[~is_terminals] += self.gamma * torch.max(self.q_model(end_states), -1)[0][~is_terminals]

        # prevent the gradient from flowing through this Tensor
        ys = ys.detach()
        # select only Q(start_state, action) and take a sum over the last dimension
        qs = torch.sum(self.q_model(start_states) * actions, -1)
        loss = torch.mean((ys - qs)**2)
        
        loss.backward()
        self.optimizer.step()
        self.eps_scheduler.tick()

        return loss.item(), torch.mean(qs).item()
        
def multi_act(ale, action, num_frames):
    """Apply `num_frames` times `action` on ale, clip and return the reward"""
    reward = 0
    # apply same action for `num_frames` frames
    for i in range(num_frames):
        reward += ale.act(action)

    #clip positive reward to 1 and negative reward to -1
    if reward > 0:
        reward = 1
    elif reward < 0:
        reward = -1

    return reward

class QEvaluator:
    def __init__(self, ale, agent, random_gen, num_episodes=10):

        # state tensors
        self.states = []

        # Initialise self.states by following random strategies
        
        
        actions = ale.getMinimalActionSet()
        
        for episode in range(10):
            frame = 0
            ale.reset_game()
            agent.reset()
            while not ale.game_over():
                # take a random action
                action_idx = agent.get_action(random_gen, epsilon=1.0)
                ale.act(actions[action_idx])

                image = preprocess(ale.getScreenGrayscale())
                # no need to pass reward or is_terminal
                agent.inform_response(action_idx, None, image, None, save_transition=False)
                frame += 1
                if frame % 10 == 0:
                    self.states.append(agent.state.copy().tensor)

        self.states = torch.stack(self.states)
    
    def evaluate(self, agent):
        return torch.mean(torch.max(agent.q_model(self.states), -1)[0])


def evaluate_reward(agent, ale, epsilon, random_gen, num_episodes=10):
    total_score = 0

    actions = ale.getMinimalActionSet()

    for episode in range(num_episodes):
        episode_score = 0
        ale.reset_game()

        agent.reset()
        while not ale.game_over():
            # Apply an action and get the resulting reward

            action_idx = agent.get_action(random_gen, epsilon)
            reward = multi_act(ale, actions[action_idx], frame_k)

            image = preprocess(ale.getScreenGrayscale())
            is_terminal = ale.game_over()
            agent.inform_response(action_idx, reward, image, is_terminal, save_transition=False)
            episode_score += reward
        
        total_score += episode_score

    return total_score / num_episodes

import sys
import pickle
from random import randrange
from ale_py import ALEInterface, SDL_SUPPORT

from matplotlib import pyplot as plt

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt("random_seed", 123)

# Load the ROM file
ale.loadROM(games[game])

# Get the list of legal actions
#legal_actions = ale.getLegalActionSet()
legal_actions = ale.getMinimalActionSet()


# image preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((110, 84)),
    transforms.ToTensor(),
])


num_actions = len(legal_actions)
print("num_actions: ", num_actions)
print(legal_actions)

if LOAD_FROM_FILE:
    with open(model_path + f"{game}_agent.pkl", "rb") as file:
        agent = pickle.load(file)
    with open(model_path + f"{game}_q_history.pkl", "rb") as file:
        q_history = pickle.load(file)
    with open(model_path + f"{game}_score_history.pkl", "rb") as file:
        score_history = pickle.load(file)
else:
    q_model = QModel(4, (110, 84), num_actions).to(DEVICE)
    optimizer = optim.RMSprop(q_model.parameters(), lr=1e-5)
    model_init_state = State(4, (110, 84))
    storage = ReplayStorage(capacity=1000000, state_size=(4, 110, 84),
                            num_actions=num_actions, batch_size=32)

    eps_scheduler = LinearScheduler(1.0, 0.1, 1000000)

    agent = Agent(q_model, optimizer, model_init_state, storage, num_actions, eps_scheduler, gamma=0.99)
    q_history = []
    score_history = []


generator = np.random.Generator(np.random.PCG64(1337))
torch.manual_seed(1337)

loss_history = []
frame_k = 3

q_evaluator = QEvaluator(ale, agent, generator)

from scipy.ndimage import uniform_filter1d
def plot_history(history, window=10, filename=None):
    plt.plot(history, label="original")
    history = np.array(history, dtype=np.float64)
    smooth_history = uniform_filter1d(history, window)
    plt.plot(smooth_history, label=f"{window} rolling average")
    plt.legend()
    if filename != None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

total_frames = agent.storage.n_added
num_training_frames = 10000000
epoch_length = 50000
epoch_used = 0

episode = 0
while total_frames < num_training_frames:
    episode += 1
    total_reward = 0
    num_frames = 0
    total_loss = 0
    total_q = 0

    ale.reset_game()

    agent.reset()

    epoch_changed = False
    while not ale.game_over():
        # Apply an action and get the resulting reward

        action_idx = agent.get_action(generator)
        reward = multi_act(ale, legal_actions[action_idx], frame_k)

        image = preprocess(ale.getScreenGrayscale())
        is_terminal = ale.game_over()
        
        agent.inform_response(action_idx, reward, image, is_terminal)
        if total_frames + num_frames > 50000:
            loss, q = agent.train_step(generator)
            total_loss += loss
            total_q += q
        total_reward += reward


        if (total_frames + num_frames) % epoch_length == 0:
            epoch_changed = True

        num_frames += 1


    if epoch_changed:
        print("epoch changed")
        q_history.append(q_evaluator.evaluate(agent).cpu().detach())
        score_history.append(evaluate_reward(agent, ale, 0.05, generator))
        plot_history(score_history,filename=f"{game}_score_history.png")
        plot_history(q_history,filename=f"{game}_q_history.png")
        plt.show()
        
    average_loss = total_loss / (num_frames + 1)
    average_q = total_q  / (num_frames + 1)
    loss_history.append(average_loss)

    print(f"Episode {episode} lasted {num_frames} train steps and ended with score {total_reward} and average loss {average_loss} and 'average q' {average_q}")
    total_frames += num_frames
    print(f"total frames: {total_frames}")


    if episode % 200 == 0 or total_frames >= num_training_frames:
        with open(model_path + f"{game}_agent.pkl", "wb") as file:
            pickle.dump(agent, file, protocol=4)
        with open(model_path + f"{game}_score_history.pkl", "wb") as file:
            pickle.dump(score_history, file, protocol=4)
        with open(model_path + f"{game}_q_history.pkl", "wb") as file:
            pickle.dump(q_history, file, protocol=4)

plot_history(score_history,filename=f"{game}_score_history.png")
plot_history(q_history,filename=f"{game}_q_history.png")

num_testing_frames = 10000

total_frames = 0

screenshot_directory = f'{game}_screenshots'

if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)
            
episode = 0
while total_frames < num_testing_frames:
    episode += 1
    total_reward = 0
    num_frames = 0

    agent.reset()
    ale.reset_game()
    while not ale.game_over():
        #screen = ale.getScreen()
        screen = ale.getScreenGrayscale()
        fig = plt.figure()
        plt.imshow(screen, interpolation='nearest')

        screenshot_file = f'{game}_{episode}_screenshot_{num_frames:05}.png'
        plt.savefig(os.path.join(screenshot_directory, screenshot_file), bbox_inches='tight')
        plt.close(fig)
        # Apply an action and get the resulting reward

        action_idx = agent.get_action(generator, epsilon=0.05)
        reward = 0
        
        # apply same action for `frame_k` frames
        for i in range(frame_k):
            reward += ale.act(legal_actions[action_idx])

        total_reward += reward

        image = preprocess(ale.getScreenGrayscale())
        is_terminal = ale.game_over()
        
        agent.inform_response(action_idx, reward, image, is_terminal, save_transition=False)
        num_frames += 1
    
    print(f"Episode {episode} lasted {num_frames} frames and ended with score {total_reward}")
    total_frames += num_frames
    print(f"total frames: {total_frames}")
