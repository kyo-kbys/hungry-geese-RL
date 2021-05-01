from kaggle_environments import make

def make_env(env_name='hungry_geese'):
    env = make(env_name, debug=True)
    return env