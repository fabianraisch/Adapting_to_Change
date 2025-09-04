import torch
import numpy as np
from collections import defaultdict
import quadprog

import torch.nn as nn

'''
Code inspired by https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py

Some parts were re-written due to a restructuring. _project_gradients function was largly copied from named repository.
'''
class GEM:

    def __init__(self, 
                 interval_count, 
                 memory_per_task=100, 
                 device='cuda',
                 loss_fn = nn.MSELoss()):
        self.model = None
        self.interval_count = interval_count
        self.memory_per_task = memory_per_task
        self.memory_data = defaultdict(list)
        self.memory_labels = defaultdict(list)
        self.loss_fn = loss_fn
        self.device = device
        self.grad_dims = []
        self.grad_size = 0
        self.current_interval = 0                       # Increase this interval in-code to use GEM properly.


    def set_model(self, model):
        """
        Post-save model to fit in our code structure
        """
        if self.model:      # In case a model was already loaded
            return

        self.model=model
        self.grad_dims = [p.numel() for p in self.model.parameters() if p.requires_grad]
        self.grad_size = sum(self.grad_dims)


    def _store_grad(self, grads, interval_id):
        offset = 0
        for p in self.model.parameters():
            if p.grad is not None:
                grad = p.grad.data.view(-1)
                grads[interval_id, offset:offset + grad.numel()] = grad
                offset += grad.numel()


    def _overwrite_grad(self, new_grad):
        offset = 0
        for p in self.model.parameters():
            if p.grad is not None:
                length = p.grad.data.numel()
                p.grad.data.copy_(new_grad[offset:offset + length].view(p.grad.data.size()))
                offset += length


    def _project_gradients(self, current_grad, mem_grads, margin=0.5, eps=1e-3):
        '''
        Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        '''
        memories_np = mem_grads.cpu().double().numpy()
        gradient_np = current_grad.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        
        return torch.Tensor(x).view(-1, 1)
    

    def observe(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        # Model must be in train in order to ensure the backprop can be done - else would throw an error.
        self.model.train()

        interval_id = self.current_interval

        # Store samples
        for i in range(x.size(0)):  # Need to do this because we have dimensions (batch_size, seq_len, feats) in the SGD scenario of ours.
            if len(self.memory_data[interval_id]) < self.memory_per_task:
                self.memory_data[interval_id].append(x[i].detach().cpu())
                self.memory_labels[interval_id].append(y[i].detach().cpu())

        # The other past gradients are stored here
        mem_grads = torch.zeros((self.interval_count, self.grad_size)).to(self.device)
        for t in range(interval_id):
            if self.memory_data[t]:
                x_prev = torch.stack(self.memory_data[t]).to(self.device)
                y_prev = torch.stack(self.memory_labels[t]).to(self.device)
                self.model.zero_grad()
                output = self.model(x_prev)
                loss = self.loss_fn(output, y_prev)
                loss.backward()                     # Propagate the loss backwards to retrieve the updated gradients.
                self._store_grad(mem_grads, t)

        # Store the current gradient(s)
        if interval_id > 0:
            current_grad = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
            # Project the gradient
            dot_prods = torch.matmul(mem_grads[:interval_id], current_grad)
            if (dot_prods < 0).sum() != 0:
                new_grad = self._project_gradients(current_grad, mem_grads[:interval_id])
                self._overwrite_grad(new_grad)