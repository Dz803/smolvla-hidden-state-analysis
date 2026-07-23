import torch
from torch import nn

from smolvla_analysis.activation_hooks import ActivationCapture, assert_hook_equivalence
from smolvla_analysis.model_inspection import resolve_pathways


class ToyPathway(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = nn.Module()
        self.text_model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.lm_expert = nn.Module()
        self.lm_expert.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

    def forward(self, value):
        for layer in self.text_model.layers:
            value = layer(value)
        for layer in self.lm_expert.layers:
            value = layer(value)
        return value


def test_activation_indexing_and_hook_equivalence():
    model = ToyPathway().eval()
    targets = resolve_pathways(model, [0.25, 1.0])
    value = torch.ones(1, 2, 4)
    result = assert_hook_equivalence(lambda: model(value), lambda: ActivationCapture(model, targets))
    assert result["max_abs_diff"] == 0
    assert result["records"] == 4

