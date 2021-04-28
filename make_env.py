from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate
from kaggle_environments.helpers import histogram
from kaggle_environments import make

def make_env(env_name='hungry_geese'):
    env = make(env_name, debug=True)
    return env