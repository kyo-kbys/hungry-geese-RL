import numpy as np

class Policy:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

class Agent:
    def __init__(self, state_dict):
        self.model = GeeseNet()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def predict(self, obs, last_obs, index):
        x = self._make_input(obs, last_obs, index)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            p, v = self.model(xt)
            
        return p.squeeze(0).detach().numpy(), v.item()
        
    # Input for Neural Network
    def _make_input(self, obs, last_obs, index):
        b = np.zeros((17, 7 * 11), dtype=np.float32)
        
        for p, pos_list in enumerate(obs.geese):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - index) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - index) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - index) % 4, pos] = 1

        # previous head position
        if last_obs is not None:
            for p, pos_list in enumerate(last_obs.geese):
                for pos in pos_list[:1]:
                    b[12 + (p - index) % 4, pos] = 1

        # food
        for pos in obs.food:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)