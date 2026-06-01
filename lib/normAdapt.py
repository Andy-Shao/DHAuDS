from copy import deepcopy
from typing import Tuple

import torch 
import torch.nn as nn

class NormAdapt(nn.Module):
    """test-time normalization adapataion"""
    def __init__(self, model: nn.Module, epsilon=1e-5, momentum=.1, reset_states=False, no_states=False) -> None:
        super().__init__()
        self.model = model
        self.model = model_formate(model=self.model, epsilon=epsilon, momentum=momentum, reset_states=reset_states, no_states=no_states)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)
    
    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

def model_formate(model: nn.Module, epsilon: float, momentum: float, reset_states: bool, no_states: bool) -> nn.Module:
    """Formate the model for the test-time normalization adaptation"""
    for component in model.modules():
        if isinstance(component, (nn.BatchNorm1d, nn.BatchNorm2d)):
            component.train()
            component.eps = epsilon
            component.momentum = momentum
            if reset_states:
                # reset the state during the test time stage
                component.reset_running_stats()
            if no_states:
                # disable state entirely and use only batch stats
                component.track_running_stats = False
                component.running_mean = None
                component.running_var = None
    return model

def get_states(model: nn.Module) -> Tuple[list, list]:
    """get the usefurl stats (batch norm stats) from the test-time normalization adaptation

    :return: property states, property names
    """
    props_states = []
    props_name = []
    for comp_name, comp in model.named_modules():
        if isinstance(comp, (nn.BatchNorm2d, nn.BatchNorm1d)):
            states = comp.state_dict()
            if comp.affine:
                # a boolean value that when set to True, this module has learnable affine parameters.
                del states['weight'], states['bias']
            for state_name, state in states.items():
                props_states.append(state)
                props_name.append(f'{comp_name}.{state_name}')
    return props_states, props_name