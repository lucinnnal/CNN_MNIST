import torch
from torch.utils.data import dataloader
import torchvision
import torchvision.transforms as transforms

def load_dataloader():
    
    # Transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Dataset & DataLoader
    trainset = torchvision.datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./Data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader