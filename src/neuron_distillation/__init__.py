"""
Workflow:
  1. Run ``clustering.py`` to produce neuron clusters from a
     trained circuit-discovery checkpoint.
  2. Run ``python -m neuron_distillation.run --config <path>`` which:
     a. Ablates clusters on both student and teacher models.
     b. Pairs clusters by importance.
     c. Trains the student with per-token CKA alignment loss.

See ``run.py`` for the single entry point.
"""
