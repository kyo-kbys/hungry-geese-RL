import os
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
from make_env import make_env
os.chdir("./agent")
from agent.policy import HungryGeese, NNAgent, MCTS

def train(game, env, agent, mcts):
    config = env.configuration
    num_episode = 1000
    print("start episode...")
    env.reset()
    obs = env.state[0].observation
    observetions = []

    # episodes
    for step in range(1):
        # print("step", step)
        # print(obs)
        # print(env.state)
        best_action = game.actions[np.argmax(mcts.getActionProb(obs, timelimit=config.actTimeout))]
        # print("actoin: ", best_action)
        # print("last obs", mcts.last_obs)
        # state_new, r, done, _ = env.step(best_action)
        # print(state_new)
        next_obs = game.getNextState(obs, mcts.last_obs, game.actions)
        observetions.append(obs)
        obs = next_obs

    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, agent])

    obs = trainer.reset()
    # print("start train")
    for _ in range(10):
        # env.render()
        action = game.actions[np.argmax(mcts.getActionProb(obs, timelimit=config.actTimeout))]
        obs, reward, done, info = trainer.step(action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        if done:
            obs = trainer.reset()


if __name__ == "__main__":
    game = HungryGeese()
    env = make_env(env_name='hungry_geese')
    env.render()
    # print(env)
    # agent = NNAgent(state_dict=None)
    # mcts = MCTS(game, agent)
    # train(game, env, agent, mcts)
    # env.run([agent, "random"])

    # Print schemas from the specification.
    # print(env.specification.observation)
    # print(env.specification.configuration)
    # print(env.specification.action)