import torch
import torch.nn as nn

def eval_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # dim=1 축으로 가장 큰 값의 값과 그리고 인덱스를 반환
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test set: {100 * correct / total}%')