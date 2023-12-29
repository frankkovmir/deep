"""
author: Frank Kovmir
frankkovmir@gmail.com
"""

import argparse
import math
import os
import torch
import random
import torch.nn as nn
from model import DeepQNetwork
from main import SpaceDodgerGame
from torch.optim import Adam
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

steps_done = 0
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def save_checkpoint(model, optimizer, episode, memory, path, steps_done):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'replay_memory': list(memory.memory),
        'steps_done': steps_done
    }
    torch.save(checkpoint, path)


def get_args():
    parser = argparse.ArgumentParser("""Deep Q Network for SpaceDodger Game""")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=int, default=1000000)
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--replay_memory_size", type=int, default=100000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.010)
    args = parser.parse_args()
    return args


def select_action(state, model, opt, device):
    global steps_done
    sample = random.random()
    eps_threshold = opt.epsilon_end + (opt.epsilon_start - opt.epsilon_end) * \
                    math.exp(-1. * steps_done / opt.epsilon_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []


# genommen von https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)

        else:
            display.display(plt.gcf())
    else:
        plt.show(block=False)


def optimize_model(opt, memory, model, target_model, optimizer, criterion, device):
    if len(memory) < opt.batch_size:
        return
    transitions = memory.sample(opt.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(opt.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * opt.gamma) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()

    return loss.item()


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 5
    output_size = 2

    writer = SummaryWriter(log_dir=opt.log_path)
    model = DeepQNetwork(input_size, output_size).to(device)
    target_model = DeepQNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    memory = ReplayMemory(opt.replay_memory_size)
    start_episode = 0

    checkpoint_path = f"{opt.saved_path}/space_dodger_checkpoint.pth"  # zahl noch dahinter
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        global steps_done
        steps_done = checkpoint.get('steps_done', 0)
        loaded_memory = checkpoint.get('replay_memory', [])
        for item in loaded_memory:
            memory.push(*item)

    log_interval = 100

    for episode in range(start_episode, opt.num_episodes):
        env = SpaceDodgerGame()
        state = env.get_state()
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)

        total_loss = 0.0
        total_reward = 0.0

        for t in count():
            action = select_action(state_tensor, model, opt, device)
            next_state, reward, done = env.step(action.item())
            next_state_tensor = None if done else torch.tensor([next_state], dtype=torch.float).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float).to(device)

            memory.push(state_tensor, action, next_state_tensor, reward_tensor)
            state_tensor = next_state_tensor

            # Optimize model
            loss = optimize_model(opt, memory, model, target_model, optimizer, criterion, device)

            if loss is not None:
                total_loss += loss

            total_reward += reward

            # Update target network
            target_net_state_dict = target_model.state_dict()
            policy_net_state_dict = model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * opt.tau + target_net_state_dict[key] * (
                            1 - opt.tau)
            target_model.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

            eps_rate = opt.epsilon_end + (opt.epsilon_start - opt.epsilon_end) * math.exp(
                -1. * steps_done / opt.epsilon_decay)

            env.render(action=action)
            env.update_spaceship_animation()
            print(
                f"Episode: {episode + 1}/{opt.num_episodes}, Iteration: {steps_done}, Level: {env.current_level}, Epsilon: {eps_rate:.4f}")

        # Log scalar values every episode
        average_loss = total_loss / (t + 1)
        writer.add_scalar('Episode_duration', t + 1, episode)
        writer.add_scalar('Epsilon_rate', eps_rate, episode)
        writer.add_scalar('Average_loss', average_loss, episode)
        writer.add_scalar('Total_reward', total_reward, episode)

        # Log histograms less frequently
        if episode % log_interval == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Episode_{episode}/{name}_grad', param.grad, episode)
                writer.add_histogram(f'Episode_{episode}/{name}_weight', param, episode)

        # Save checkpoint
        if episode % 100 == 0 and episode != 0:
            save_checkpoint_path = f"{opt.saved_path}/space_dodger_checkpoint_{episode}.pth"
            save_checkpoint(model, optimizer, episode, memory, save_checkpoint_path, steps_done)



    torch.save(model.state_dict(), f"{opt.saved_path}/space_dodger_final.pth")

    writer.close()
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
