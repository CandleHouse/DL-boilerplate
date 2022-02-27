import torch


class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, pred, true):
        error = torch.subtract(pred, true)
        mape = torch.mean(torch.abs(torch.divide(error, true)))
        loss = torch.multiply(mape, 100)
        return loss