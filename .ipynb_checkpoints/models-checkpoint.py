import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet152': [resnet152, 2048]
}


class SupConResNet(nn.Module):
    """encoder + projection head"""
    def __init__(self, encoder='resnet50', head='mlp', load_pt_encoder=False, feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[encoder]
        self.dim_in = dim_in
        self.feat_dim = feat_dim
        self.encoder = model_fun() if not load_pt_encoder else model_fun(pretrained=True)
        self.proto = None
        # remove the last fc layer
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        
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

    def encode(self, x):
        """return the features encoded by resnet of size (B, D_in)"""
        x = self.encoder(x)
        while x.dim() > 2:
            x = torch.squeeze(x, dim=2)
        return x
    
    def project(self, x):
        """return the features encoded by resnet and projection head of size (B, D_ft)"""
        x = self.encode(x)
        x = self.head(x)
        return x
    
    def forward(self, x):
        """return the normalized features encoded by resnet and projection head of size (B, D_ft)"""
        x = self.project(x)
        x = F.normalize(x, dim=1) # L2 normalization over each feature vector
        return x
    
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


class LMCosResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, encoder='resnet50', head=None, load_pt_encoder=False, num_classes=17):
        super(LMCLResNet, self).__init__()
        model_fun, dim_in = model_dict[encoder]
        self.encoder = model_fun() if not load_pt_encoder else model_fun(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        
        self.weight = Parameter(torch.Tensor(dim_in, num_classes))
        nn.init.xavier_uniform_(self.weight)
    
    def encode(self, x):
        """return the features encoded by resnet of size (B, D_in)"""
        x = self.encoder(x)
        while x.dim() > 2:
            x = torch.squeeze(x, dim=2)
        return x
    
    def forward(self, x):
        """return the cosine similarity between each feature and class learnable weights of size (B, C)"""
        x = self.encode(x) # (B, D_in)
        logits = cosine_sim(x, self.weight) # (B, C)
        return logits
    
#     def forward(self, x, y):
#         x = self.encode(x)
#         output = self.classifier(x, y)
#         return output


        
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)


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


    
