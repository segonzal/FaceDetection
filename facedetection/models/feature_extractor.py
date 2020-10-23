import torch.nn as nn
from collections import OrderedDict


class FeatureExtractor(nn.ModuleDict):
    def __init__(self, model, return_layers):
        layer_set = set(return_layers)
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in layer_set:
                layer_set.remove(name)
            if not layer_set:
                break
        super(FeatureExtractor, self).__init__(layers)
        self.return_layers = return_layers

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)
        return outputs
