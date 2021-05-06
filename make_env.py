from kaggle_environments import make

def make_env(env_name):
    env = make(env_name, debug=True)
    return env

def make_trainer(env_name='hungry_geese', players=["random"]):
    env = make_env(env_name)
    # Make trainer for kaggle env agent
    trainer = env.train([None] + players)
    return trainer