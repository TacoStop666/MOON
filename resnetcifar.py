'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation) 


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1 # The expansion factor for the BasicBlock is 1, meaning output channels = input channels × 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__() 
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d # Default normalization layer is BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock") 
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) # 3x3 convolution with stride
        self.bn1 = norm_layer(planes) # Normalization layer for the first convolution
        self.relu = nn.ReLU(inplace=True) # ReLU activation function
        self.conv2 = conv3x3(planes, planes) # Second 3x3 convolution
        self.bn2 = norm_layer(planes) # Normalization layer for the second convolution
        self.downsample = downsample # Downsample layer if needed
        # downsample: a layer (or sequence of layers) used to match the dimensions of the input (identity) to the output of the residual block when they are different
        self.stride = stride # Stride for the block

    def forward(self, x):
        identity = x # Save the input for the skip connection

        out = self.conv1(x) # First convolution
        out = self.bn1(out) # Apply normalization
        out = self.relu(out) # ReLU activation

        out = self.conv2(out) # Second convolution
        out = self.bn2(out) # Apply normalization

        if self.downsample is not None:
            identity = self.downsample(x) # Downsample the input if needed

        out += identity # Add the (possibly downsampled) input to the output (skip connection)
        out = self.relu(out) # ReLU activation after the skip connection

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # The expansion factor for the Bottleneck block is 4, meaning the output channels are four times the input channels.

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # Default normalization layer is BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups # Calculate the width of the bottleneck block based on the base width and number of groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) # 1x1 convolution to reduce the number of channels
        self.bn1 = norm_layer(width) # Normalization layer for the first convolution
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # 3x3 convolution with stride, groups, and dilation
        self.bn2 = norm_layer(width) # Normalization layer for the second convolution
        self.conv3 = conv1x1(width, planes * self.expansion) # 1x1 convolution to increase the number of channels
        self.bn3 = norm_layer(planes * self.expansion) # Normalization layer for the third convolution
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # First 1x1 convolution
        out = self.bn1(out) # Apply normalization
        out = self.relu(out) # ReLU activation
 
        out = self.conv2(out) # Second 3x3 convolution
        out = self.bn2(out) # Apply normalization
        out = self.relu(out) # ReLU activation

        out = self.conv3(out) # Third 1x1 convolution
        out = self.bn3(out) # Apply normalization

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar10(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1 # Dilation factor for the convolutions, used for dilated convolutions
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups # Number of groups for group convolution
        self.base_width = width_per_group # Base width for the bottleneck block
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False) # Initial convolution layer with 3x3 kernel, stride 1, and padding 1
        self.bn1 = norm_layer(self.inplanes) # Normalization layer for the initial convolution
        self.relu = nn.ReLU(inplace=True) 
        self.layer1 = self._make_layer(block, 64, layers[0]) # First layer of the ResNet, using the specified block type and number of layers
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0]) # Second layer with stride 2 and possible dilation
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) # Third layer with stride 2 and possible dilation
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]) # Fourth layer with stride 2 and possible dilation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Adaptive average pooling to output a fixed size (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes) # Fully connected layer to output the final class scores

        for m in self.modules():
            if isinstance(m, nn.Conv2d): # Initialize convolutional layers 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Initialize convolutional layers with He initialization
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # Initialize normalization layers
                nn.init.constant_(m.weight, 1) # Set the weight to 1
                nn.init.constant_(m.bias, 0) # Set the bias to 0

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18_cifar10(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)



def ResNet50_cifar10(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], **kwargs)