from copy import deepcopy
from torch import nn
import torch

'''
Code originated from https://github.com/moskomule/ewc.pytorch
Adapted to work for timeseries instead of multiclass-classificaiton task.
'''


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return t


class EWC(object):
    def __init__(self, 
                 model: nn.Module, 
                 dataset: list, 
                 use_cuda=True,
                 is_disabled=False,    # Indicactes if this EWC penalty should be 0
                 loss_function = nn.MSELoss()):
        """
        dataset: list of (input, target) tuples
        """
        self.model = model
        self.dataset = dataset
        self.use_cuda = use_cuda
        self.loss_function = loss_function
        self.is_disabled = is_disabled

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.clone().detach().zero_()

        self.model.train()
        for input, output in self.dataset:
            self.model.zero_grad()
            input = variable(input, use_cuda=self.use_cuda)
            output = variable(output, use_cuda=self.use_cuda)

            # Unsqueeze because batch dimension is missing here and we use batch dimension in our models.
            input = input.unsqueeze(0)

            model_output = self.model(input).squeeze()
            loss = self.loss_function(model_output, output)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += (p.grad.data ** 2) / len(self.dataset)

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0

        if self.is_disabled:    # In this case, EWC is basically a no-op.
            return loss

        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss