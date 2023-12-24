import argparse
import torch
from main import SpaceDodgerGame
import numpy as np
import pygame
from model import DeepQNetwork
import os

def get_args():
    parser = argparse.ArgumentParser("""Test Deep Q Network on SpaceDodger Game""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args

def test(opt):
    input_size = 10
    output_size = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork(input_size, output_size).to(device)

    model_path = f"{opt.saved_path}/space_dodger_final.pth"
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
    else:
        print("No saved model found. Please check the path.")
        return

    model.eval()
    game = SpaceDodgerGame()
    state = game.reset()

    while True:
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)

        with torch.no_grad():
            prediction = model(state_tensor)
        action = torch.argmax(prediction).item()

        next_state, _, done = game.step(action)

        state = next_state

        pygame.display.flip()
        game.render(action)

        if done:
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
