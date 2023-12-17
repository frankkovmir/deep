import torch
from main import ModifiedPirateBayGame
from model import DeepQNetwork
from utils import pre_processing

image_size = 84

# Load the trained model
model = DeepQNetwork()
model.load_state_dict(torch.load('dqn_model.pth'))
model.eval()

env = ModifiedPirateBayGame()
state = torch.tensor([pre_processing(env, image_size, image_size)], dtype=torch.float)

while True:
    with torch.no_grad():
        action = model(state).max(1)[1].view(1, 1)
    next_state, _, done = env.play_step_with_agent(action.item())

    next_state = torch.tensor([pre_processing(next_state, image_size, image_size)], dtype=torch.float)

    state = next_state

    if done:
        state = env.reset()
        state = torch.tensor([pre_processing(state, image_size, image_size)], dtype=torch.float)
