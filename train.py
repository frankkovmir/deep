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
from utils import pre_processing




def process_frame(frame, image_size):
    processed_frame = pre_processing(frame, image_size, image_size)
    return np.squeeze(processed_frame)

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network for SpaceDodger Game""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=50000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args

def stack_frames(batch_frames, image_size):
    stacked_frames = []
    for frame_set in batch_frames:
        # Stack 4 frames for each state in the batch
        stacked = np.stack([process_frame(frame, image_size) for frame in frame_set], axis=0)
        stacked_frames.append(stacked)
    return np.array(stacked_frames)


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.MSELoss()
    replay_memory = []
    scores = []
    mean_scores = []
    total_score = 0

    for episode in range(opt.num_episodes):
        env = SpaceDodgerGame()
        state = env.reset()
        state_processed = process_frame(state, opt.image_size)
        state_stack = [state_processed] * 4  # Initialize the state stack
        score = 0
        done = False
        iter_count = 0  # Track the number of iterations within an episode
        while not done:
            epsilon = opt.epsilon_end + (opt.epsilon_start - opt.epsilon_end) * \
                      (1 - episode / opt.num_episodes)
            if np.random.rand() <= epsilon:
                action = randint(0, 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor([state_stack], dtype=torch.float)
                    action = model(state_tensor).max(1)[1].view(1, 1).item()

            next_state, reward, done = env.step(action)
            next_state_processed = process_frame(next_state, opt.image_size)

            # Update state stack
            next_state_stack = state_stack[1:] + [next_state_processed]

            score += reward
            replay_memory.append((state_stack, action, reward, next_state_stack, done))

            state_stack = next_state_stack  # Update the state stack

            if len(replay_memory) > opt.replay_memory_size:
                del replay_memory[0]

            if len(replay_memory) > opt.batch_size:
                batch = sample(replay_memory, opt.batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

                state_batch = torch.tensor(stack_frames(state_batch, opt.image_size), dtype=torch.float)
                next_state_batch = torch.tensor(stack_frames(next_state_batch, opt.image_size), dtype=torch.float)

                action_batch = torch.tensor(action_batch).long()
                reward_batch = torch.tensor(reward_batch)
                terminal_batch = torch.tensor(terminal_batch, dtype=torch.float)

                current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_state_batch).max(1)[0]
                expected_q_values = reward_batch + opt.gamma * next_q_values * (1 - terminal_batch)

                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss', loss.item(), episode)
                writer.add_scalar('Epsilon', epsilon, episode)

            env.render()
            # Print training progress
            print("Episode: {}/{}, Iteration: {}, Points: {}, Epsilon: {:.4f}".format(
                episode + 1, opt.num_episodes, iter_count, env.points, epsilon))

            iter_count += 1  # Increment the iteration count

        scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)
        writer.add_scalar('Score', score, episode)

        if episode % 100 == 0 and episode != 0:
            torch.save(model.state_dict(), f"{opt.saved_path}/space_dodger_{episode}.pth")

    torch.save(model.state_dict(), f"{opt.saved_path}/space_dodger_final.pth")
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)