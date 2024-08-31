import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import numpy as np

from neural_network import MarioNet
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Mario:

    def __init__(self, state_dim, action_dim, dir_name, distributed_training=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dir_name = dir_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        """
        if we are loading the model then instead of instantiating a new MarioNet we will load the model in case the model has changed
        since the last time it wa
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if os.path.exists(os.path.join(dir_name, 'model.mdl')):
            self.net = torch.load(os.path.join(dir_name, 'model.mdl'))
        else:
            self.net = MarioNet(self.state_dim, self.action_dim).float()
            torch.save(self.net, os.path.join(dir_name, 'model.mdl'))

        if distributed_training:
            self.net = DDP(self.net, device_ids=[0])
        else:
            self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        # self.exploration_rate_decay = 0.999999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # self.save_every = 5e5  # no. of experiences between saving Mario Net ORIGINAL
        self.save_every = 1e5  # no. of experiences between saving Mario Net

        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.load()

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(
                state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(
            TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done},
                       batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in
                                                   ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """
        Saves the weights of the online and target networks to a file in the checkpoints directory
        """
        if not os.path.exists(os.path.join(self.dir_name, "checkpoints")):
            os.makedirs(os.path.join(self.dir_name, "checkpoints"))
            
        max_checkpoint = 0
        for checkpoint in os.listdir(os.path.join(self.dir_name, "checkpoints")):
            if checkpoint.endswith(".chkpt"):
                i = int(checkpoint.replace("mario_net_", "").replace(".chkpt", ""))
                if i >= max_checkpoint:
                    max_checkpoint = i
                
        save_path = os.path.join(self.dir_name, "checkpoints", f"mario_net_{max_checkpoint+1}.chkpt")

        torch.save(
            {"online": self.net.online.state_dict(),
             "target": self.net.target.state_dict(),
             "exploration_rate": self.exploration_rate,
             "curr_step": self.curr_step},
            save_path)

        with open(os.path.join(self.dir_name, 'steps.txt'), 'a') as f:
            s = f"mario_net_{i}.chkpt steps:{self.curr_step}\n"
            f.write(s)

        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self):
        """
        Load the weights of the online and target networks from the most recent checkpoint file in the checkpoints directory
        """
        if not os.path.exists(os.path.join(self.dir_name, 'checkpoints')):
            os.makedirs(os.path.join(self.dir_name, 'checkpoints'))

        max_checkpoint = 0
        for checkpoint in os.listdir(os.path.join(self.dir_name, "checkpoints")):
            if checkpoint.endswith(".chkpt"):
                i = int(checkpoint.replace("mario_net_", "").replace(".chkpt", ""))
                if i >= max_checkpoint:
                    max_checkpoint = i

        if max_checkpoint == 0:
            print("No checkpoints found")
            return
                
        load_path = os.path.join(self.dir_name, "checkpoints", f"mario_net_{max_checkpoint}.chkpt")

        checkpoint = torch.load(load_path)
        self.net.online.load_state_dict(checkpoint["online"])
        self.net.target.load_state_dict(checkpoint["target"])
        self.exploration_rate = checkpoint["exploration_rate"]
        self.curr_step = checkpoint["curr_step"]
        print(f'MarioNet loaded from {load_path}')

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
