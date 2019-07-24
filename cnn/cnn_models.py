import torch
import torchvision

def Alexnet(num_classes=17, pretrained=True):
    alexnet = torchvision.models.alexnet(pretrained=pretrained)
    alexnet.classifier._modules['6'] = torch.nn.Linear(4096, num_classes)
    return alexnet

def Vgg(num_classes=17, pretrained=True):
    vgg = torchvision.models.vgg(pretrained=pretrained)
    vgg.classifier._modules['6'] = torch.nn.Linear(4096, num_classes)
    return vgg