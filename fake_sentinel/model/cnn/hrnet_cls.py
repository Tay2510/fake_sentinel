import torch
import torch.nn as nn
import torch.nn.functional as F

from fake_sentinel.model.cnn.hrnet_landmark import HighResolutionNet, Bottleneck
from fake_sentinel.model.cnn.hrnet_configs import HRNET_CONFIGS
from fake_sentinel.data.paths import AFLW_PRETRAINED_HRNET as PRETRAINED_WEIGHT_PATH

CLS_BN_MOMENTUM = 0.1


class HRNetClassifier(HighResolutionNet):
    def __init__(self, pretrained=True):
        super(HRNetClassifier, self).__init__(HRNET_CONFIGS)

        if pretrained:
            super(HRNetClassifier, self).init_weights(PRETRAINED_WEIGHT_PATH)

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.pre_stage_channels)

        self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, momentum=CLS_BN_MOMENTUM), nn.ReLU(inplace=True))

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(in_channels=head_channels[3] * head_block.expansion, out_channels=2048, kernel_size=1, stride=1,
                padding=0), nn.BatchNorm2d(2048, momentum=CLS_BN_MOMENTUM), nn.ReLU(inplace=True))

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


def hrnet_classifier(pretrained=True):
    model = HRNetClassifier(pretrained=pretrained)

    return model
