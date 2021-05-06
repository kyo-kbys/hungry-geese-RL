from kaggle_environments import make

def make_env(env_name):
    env = make(env_name, debug=True)
    return env

def make_trainer(env_name='hungry_geese', enemy_players=["random"]):
    env = make_env(env_name)
    # Make trainer for kaggle env agent
    if enemy_players is None:
        trainer = env.train([None]) # Only agent
    else:
        trainer = env.train([None] + players) # with enemy
    return trainer