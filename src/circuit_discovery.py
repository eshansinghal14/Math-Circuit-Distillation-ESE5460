from circuit_discovery import (  # noqa: F401
    ProblemEncoder,
    ActivationsEncoder,
    ProblemClassifier,
    NeuronMask,
    CircuitDiscoveryModel,
    CircuitLoss,
    parse_equation,
    _stack_layer_activations,
    train_circuit_discovery,
)


if __name__ == "__main__":
    train_circuit_discovery(k_classes=8, epochs=20000, resume_model='model_3500')