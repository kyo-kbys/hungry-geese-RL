import os
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
from make_env import make_trainer
import torch
os.chdir("./agent")
from agent.policy import RolloutStrage, GNet, Brain

GAMMA = 0.99
MAX_STEPS = 150
NUM_EPISODES = 1000
NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5

value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

def train(trainer, actor_critic, brain, rollouts):
    each_step = 0
    episode_reward = 0
    final_reward = 0

    for episode in range(NUM_EPISODES):
        index = 0
        # Reset observation
        obs = trainer.reset()
        last_obs = None

        for step in range(NUM_ADVANCED_STEP):
            # Get action
            action = actor_critic.get_action(obs, last_obs, index)
            # Get objects
            next_obs, reward, done, info = trainer.step(action)

            if done:
                print(f"{episode} Episode: Finish after {step} steps")
                episode += 1

                # Add reward
                if each_step < MAX_STEPS:
                    reward -= 10.0
                else:
                    reward += 10.0

                each_step = 0 # Reset step

            else:
                each_step += 1

            # Convert into torch
            reward = torch.FloatTensor([reward]).float()
            episode_reward += reward

            # Get mask
            mask = torch.FloatTensor([0.0] if done else [1.0])

            # Final reward
            final_reward *= mask
            final_reward += (1 - mask) * episode_reward
            episode_reward *= mask

            # Insert current step transition into memory
            rollouts.insert(obs, last_obs, action, reward, mask)

            if done:
                print("observation reset")
                obs = trainer.reset() # Reset trainer (env)
            else:
                obs = next_obs

        '''
        Update Network
        '''
        with torch.no_grad():
            next_value = actor_critic.get_values(rollouts.observations[-1], last_obs, index)
        # Get returns
        rollouts.compute_returns(next_value)
        # Update
        brain.update(rollouts)
        rollouts.after_update()

if __name__ == "__main__":
    trainer = make_trainer(env_name='hungry_geese', enemy_players=None)
    actor_critic = GNet()
    brain = Brain(actor_critic)
    rollouts = RolloutStrage(NUM_ADVANCED_STEP)
    train(trainer, actor_critic, brain, rollouts)