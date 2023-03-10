"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import MarginCosineProduct

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, encoder='resnet50', head='mlp', load_pt_encoder=False, feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[encoder]
        self.dim_in = dim_in
        self.feat_dim = feat_dim
        self.encoder = model_fun()
        self.proto = None
        if encoder == 'resnet50':
            from torchvision.models import resnet50
            print("[Model] use encoder structure from PyTorch")
            model = resnet50()
            self.encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
        if load_pt_encoder:
            print("[Model] use pretrained encoder from PyTorch")
            from torchvision.models import resnet50
            if encoder == 'resnet50':
                model = resnet50(pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        while feat.dim() > 2:
            feat = torch.squeeze(feat, dim=2)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
    def return_proto(self, data_loader, num_classes=17):
        if self.proto is not None:
            return self.proto
        self.proto = torch.zeros(size=[num_classes, self.dim_in])
#         self.proto = torch.zeros(size=[num_classes, self.feat_dim])
        features = []
        labels = []
        for idx, (imgs, lbs) in enumerate(data_loader):
            imgs = imgs.float().cuda()
            lbs = lbs.cuda()
            feats = self.encoder(imgs)
            while feats.dim() > 2:
                feats = torch.squeeze(feats, dim=2)
#             feats = self.head(feats)
            features.append(feats)
            labels.append(lbs)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()
        for i in range(num_classes):
            self.proto[i] = features[labels==i].mean(dim=0)
        return self.proto


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LMCLResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, encoder='resnet50', load_pt_encoder=False, num_classes=17):
        super(LMCLResNet, self).__init__()
        model_fun, dim_in = model_dict[encoder]
        self.encoder = model_fun()
        
        if load_pt_encoder:
            print("[Model] use pretrained encoder from PyTorch")
            from torchvision.models import resnet50
            if encoder == 'resnet50':
                model = resnet50(pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
            
        self.layer = MarginCosineProduct(dim_in, num_classes)
    
    def forward(self, x, y):
        output = self.encoder(x)
        output = self.layer(torch.squeeze(output), y)
        return output

class MCPClassifier(nn.Module):
    """Margin Cosine Loss Classifier"""
    def __init__(self, in_features, num_classes=17, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)

        return output

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, encoder='resnet50', num_classes=17):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[encoder]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, name='resnet50', num_classes=17, projection_size=4096):
        super(MLPClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.mlp = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, projection_size)
    )
        self.fc = nn.Linear(projection_size, num_classes)
    
    def forward(self, features):
        return self.fc(self.mlp(features))

class SimSiamClassifier(nn.Module):
    """SimSiam classifier"""
    def __init__(self, name='resnet50', num_classes=17, projection_size=4096):
        super(SimSiamClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.mlp = nn.Sequential(
        nn.Linear(feat_dim, feat_dim, bias=False),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, feat_dim, bias=False),
        nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
        self.fc = nn.Linear(projection_size, num_classes)
    
    def forward(self, features):
        return self.fc(self.mlp(features))



