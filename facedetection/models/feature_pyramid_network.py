import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, extra_num=2):
        super(FeaturePyramidNetwork, self).__init__()

        self.stage_num = len(in_channels)
        self.extra_num = extra_num

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.extra_blocks = nn.ModuleList()

        for i in range(self.stage_num):
            inner_module = nn.Conv2d(in_channels[i], out_channels, 3)
            layer_module = nn.Conv2d(out_channels, out_channels, 1, padding=1)
            self.inner_blocks.append(inner_module)
            self.layer_blocks.append(layer_module)

        oc = in_channels[-1]
        for _ in range(self.extra_num):
            extra_module = nn.Conv2d(oc, out_channels, 3, 2, 1)
            oc = out_channels
            self.extra_blocks.append(extra_module)

        self.init_weights()

    def forward(self, inputs):
        assert self.stage_num == len(inputs), \
            'FPN requires a list of {} features from a backbone, got {}.' % (self.stage_num, len(inputs))

        outputs = []
        for i in range(self.stage_num-1, -1, -1):
            p = self.inner_blocks[i](inputs[i])
            if outputs:
                p = p + F.interpolate(outputs[0], size=p.shape[-2:], mode='nearest')
            p = self.layer_blocks[i](p)
            outputs.insert(0, p)

        p = inputs[-1]
        for i in range(self.extra_num):
            p = self.extra_blocks[i](p)
            outputs.append(p)
            p = F.relu(p)

        return outputs

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
