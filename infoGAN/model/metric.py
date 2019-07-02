import torch


def prediction_acc(output, target):
    with torch.no_grad():
        pred = torch.round(output)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)
