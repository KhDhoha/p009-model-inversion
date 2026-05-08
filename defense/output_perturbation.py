import torch
import torch.nn as nn


class OutputPerturbationDefense(nn.Module):
    """
    Post-training defense that adds perturbations to model predictions
    to defend against model inversion attacks.
    """

    def __init__(self, model, perturbation_strength=0.1):
       
        super().__init__()

        self.model = model
        self.perturbation_strength = perturbation_strength

    def forward(self, x):
        """
        forward pass with output perturbation.
        """

        # Original logits from classifier
        logits = self.model(x)

        # Add noise only during inference
        if not self.training:
            noise = (
                torch.randn_like(logits)
                * self.perturbation_strength
            )

            logits = logits + noise

        return logits