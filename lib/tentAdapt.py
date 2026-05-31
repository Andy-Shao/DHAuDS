from copy import deepcopy
from typing import Tuple, Any, Dict

import torch 
import torch.nn as nn
import torch.jit
import torch.optim as optim

class TentAdapt(nn.Module):
    """Tent test-time adaptation"""
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, steps=1, resetable=False) -> None:
        super().__init__()
        self.model = model_formate(model=model)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.resetable = resetable

        self.model_states, self.optimizer_states = copy_state_in_model_optimizer(model=self.model, optimizer=self.optimizer)

    def forward(self, x: torch.Tensor, adapted_fn=None) -> torch.Tensor:
        if self.resetable:
            self.reset()
        
        for _ in range(self.steps):
            if adapted_fn is None:
                outputs = adapted_forward(x=x, model=self.model, optimizer=self.optimizer)
            else:
                outputs = adapted_fn(x=x, model=self.model, optimizer=self.optimizer)
        
        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_states is None:
            raise Exception('Without saved model/optimizer cannot be reset')
        load_model_optimizer_state(self.model, self.optimizer, self.model_states, self.optimizer_states)

def copy_state_in_model_optimizer(model: nn.Module, optimizer: optim.Optimizer) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_states = deepcopy(model.state_dict())
    optimizer_states = deepcopy(optimizer.state_dict())
    return model_states, optimizer_states

def load_model_optimizer_state(
        model: nn.Module, optimizer: optim.Optimizer, model_states: Dict[str, Any], optimizer_states: Dict[str, Any]
    ) -> None:
    model.load_state_dict(model_states)
    optimizer.load_state_dict(optimizer_states)

@torch.enable_grad()
def adapted_forward(x: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer) -> torch.Tensor:
    outputs = model(x)
    optimizer.zero_grad()
    loss = adapted_loss_fn(outputs).mean(0)
    loss.backward()
    optimizer.step()

    return outputs

@torch.jit.script
def adapted_loss_fn(x: torch.Tensor) -> torch.Tensor:
    # negative entropy function
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def model_formate(model: nn.Module) -> nn.Module:
    """Formate the model for tent test-time adaptation"""
    model.train()
    model.requires_grad_(False)
    for component in model.modules():
        if isinstance(component, (nn.BatchNorm1d, nn.BatchNorm2d)):
            component.requires_grad_(True)
            # force use of batch stats in train and eval modes
            component.track_running_stats = False
            component.running_mean = None
            component.running_var = None
    return model

def get_params(model: nn.Module) -> Tuple[list, list]:
    """Collect the affine weight(scale) + bias(shift) parameters from batch norms.""" 
    params = []
    param_names = []
    for component_name, component in model.named_modules():
        if isinstance(component, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param_name, param in component.named_parameters():
                if param_name in ['weight', 'bias']:
                    params.append(param)
                    param_names.append(f'{component_name}.{param_name}')
    return params, param_names

def verify_model(model: nn.Module) -> None:
    """Check model for compatability with tent."""
    is_training_stage = model.training
    assert is_training_stage, 'tent model should be in the training status'
    param_grad_status = [p.requires_grad for p in model.parameters()]
    any_param_can_grad = any(param_grad_status)
    all_params_can_grad = not all(param_grad_status)
    assert any_param_can_grad, 'tent model requires some parameters to grade'
    assert all_params_can_grad, 'tent model does not allow all parameters to grade during test-time adaptation stage'
    include_batchNorm = any([isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) for m in model.modules()])
    assert include_batchNorm, 'tent requires the batchNorm structure'