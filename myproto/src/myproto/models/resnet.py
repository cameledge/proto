import torchvision.models as tv
import torch.nn as nn

def build_resnet18(num_classes=10, pretrained=False):
    model = tv.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
