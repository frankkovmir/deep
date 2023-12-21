import argparse
import torch
from main import SpaceDodgerGame
from utils import pre_processing
import numpy as np
import pygame
from model import DeepQNetwork
import os


def get_args():
    parser = argparse.ArgumentParser("""Test Deep Q Network on SpaceDodger Game""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def process_frame(frame, image_size):
    processed_frame = pre_processing(frame, image_size, image_size)
    return np.squeeze(processed_frame)

def test(opt):
    model = DeepQNetwork()

    model_path = f"{opt.saved_path}/space_dodger_final.pth"
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    else:
        print("No saved model found. Please check the path.")
        return

    model.eval()
    game = SpaceDodgerGame()
    state = game.reset()
    state_processed = process_frame(state, opt.image_size)
    state_stack = [state_processed] * 4  # Initialize the state stack
    clock = pygame.time.Clock()  # Create a clock object
    fps = 30

    while True:
        state_tensor = torch.tensor([state_stack], dtype=torch.float)
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()

        prediction = model(state_tensor)[0]
        action = torch.argmax(prediction).item()

        next_state, _, done = game.step(action)
        next_state_processed = process_frame(next_state, opt.image_size)

        next_state_stack = state_stack[1:] + [next_state_processed]
        state_stack = next_state_stack  # Update the state stack
        clock.tick(fps)

        game.render()

        if done:
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
