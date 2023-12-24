import argparse
import os
import shutil
from random import randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from model import DeepQNetwork
from main import SpaceDodgerGame
from torch.optim import Adam
from collections import deque

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network for SpaceDodger Game""")
    parser.add_argument("--batch_size", type=int, default=180, help="The number of images per batch")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--replay_memory_size", type=int, default=10000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--target_update", type=int, default=10, help="Number of episodes to update the target network")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 10
    output_size = 3
    model = DeepQNetwork(input_size, output_size).to(device)
    target_model = DeepQNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    global_step = 0
    start_episode = 0
    replay_memory = deque(maxlen=opt.replay_memory_size)
    epsilon_decay_rate = (opt.epsilon_start - opt.epsilon_end) / opt.num_episodes


    checkpoint_path = f"{opt.saved_path}/space_dodger_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        replay_memory = checkpoint.get('replay_memory', [])
        print(f"Resuming training from episode {start_episode}")

    for episode in range(start_episode, opt.num_episodes):
        env = SpaceDodgerGame()
        state = env.get_state()
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)
        episode_rewards = 0
        done = False
        iter_count = 0

        while not done:
            epsilon = max(opt.epsilon_end, opt.epsilon_start - epsilon_decay_rate * episode)
            q_values = model(state_tensor)
            action = randint(0, output_size - 1) if np.random.rand() <= epsilon else torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float).to(device)
            replay_memory.append((state_tensor, action, reward, next_state_tensor, done))
            state_tensor = next_state_tensor
            episode_rewards += reward

            # state prints
            # spaceship_pos = next_state[0:2]
            # asteroids_info = next_state[2:4]
            # print(f"Spaceship Position: {spaceship_pos}")
            # print(f"Asteroids: {asteroids_info}")
            # print(f"Reward: {reward}, Done: {done}")
            # print(f"State Tensor: {state_tensor}")
            #print(f"Stored in replay memory: State: {next_state}, Action: {action}, Reward: {reward}, Done: {done}")

            if len(replay_memory) > opt.replay_memory_size:
                del replay_memory[0]

            if len(replay_memory) >= opt.batch_size:
                batch = sample(replay_memory, opt.batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

                state_batch = torch.stack(state_batch).to(device)
                next_state_batch = torch.stack(next_state_batch).to(device)
                action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
                terminal_batch = torch.tensor(terminal_batch, dtype=torch.float).to(device)

                current_q_values = model(state_batch).squeeze(1)
                action_indices = action_batch.unsqueeze(-1)
                next_q_values = target_model(next_state_batch).squeeze(1)
                #print(f"Shape of next_q_values: {next_q_values.shape}")
                #print(f"Shape of action_indices: {action_indices.shape}")
                max_next_q_values = next_q_values.max(1)[0]
                reward_batch = reward_batch.unsqueeze(1)
                terminal_batch = terminal_batch.unsqueeze(1)
                #print(f'current_q_values: {current_q_values}')
                #print(f"Shape of next_q_values: {next_q_values.shape}")
                #print(f'next_q_values: {next_q_values}')
                #print(f'max_next_q_values: {max_next_q_values}')
                #print(f'reward_batch: {reward_batch}')
                #print(f'terminal_batch: {terminal_batch}')
                max_next_q_values = max_next_q_values.unsqueeze(1)
                expected_q_values = reward_batch + (opt.gamma * max_next_q_values * (1 - terminal_batch))


                # Reshape action_batch to use as index

                #print(f'action_indices: {action_indices}')

                current_q_actions = current_q_values.gather(1, action_indices)
                #print(f'current_q_actions: {current_q_actions}')

                # print(f'max_next_q_values shape: {max_next_q_values.shape}')
                # print(f'expected_q_values shape: {expected_q_values.shape}')
                # print(f'current_q_actions shape: {current_q_actions.shape}')
                # Calculate loss
                loss = criterion(current_q_actions, expected_q_values)
                #print(f"Training Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
                optimizer.step()

                if episode % opt.target_update == 0:
                    target_model.load_state_dict(model.state_dict())

                # TensorBoard logging
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                writer.add_scalar('Q-Values/Average', q_values.mean().item(), global_step)
                writer.add_histogram('Q-Values/Distribution', q_values.detach().cpu().numpy(), global_step)
                writer.add_histogram('Actions/Distribution', action_batch.detach().cpu().numpy(), global_step)
                writer.add_scalar('Performance/Total Reward', episode_rewards, episode)
                writer.add_scalar('Performance/Epsilon', epsilon, episode)
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Weights/{name}', param, global_step)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step)


            env.render(action=action)
            env.update_spaceship_animation()
            # Print training progress
            print("Episode: {}/{}, Iteration: {}, Level: {}, Reward: {}, Epsilon: {:.4f}".format(
                episode + 1, opt.num_episodes, iter_count, env.current_level, reward, epsilon))
            global_step += 1
            iter_count += 1

        if episode % 100 == 0 and episode != 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'replay_memory': replay_memory
            }
            torch.save(checkpoint, checkpoint_path)

    torch.save(model.state_dict(), f"{opt.saved_path}/space_dodger_final.pth")
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)