from .models import (
    ProblemEncoder,
    ProblemClassifier,
    NeuronMask,
    CircuitDiscoveryModel,
    CircuitLoss,
)
from .utils import parse_equation, _stack_layer_activations


def train_circuit_discovery(*args, **kwargs):
    from .main import train_circuit_discovery as _train
    return _train(*args, **kwargs)
