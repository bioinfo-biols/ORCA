import torch

import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        ch = 56
        self.rnns = torch.nn.ModuleList([
            torch.nn.LSTM(ch, ch, 1, bidirectional=False),
            torch.nn.LSTM(ch, 128, 2, bidirectional=False)
        ])

    def forward(self, x):
        x = x.permute((1,0,2))
        for rnn in self.rnns:
            x, _ = rnn(x)
            x = x.flip([0])
        x = x.permute((1,2,0))
        x = x.reshape(x.size(0), 128 * 5)
        return x



class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        self.fc1 = nn.Linear(128* 5, 128
                            )
        self.fc2 = nn.Linear(128, 128
                            )
        self.fc3 = nn.Linear(128, 2)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, input):
        logits = F.relu(self.fc1(input))
        logits = self.fc2(F.dropout(logits))
        logits1 = F.relu(logits)
        logits_ce = self.fc3(logits1)
        logits_mse = self.fc4(logits1)
        ce = F.log_softmax(logits_ce, 1)
        mse = logits_mse[:,0]
        
        return ce,mse
    
class Domain_classifier(nn.Module):

    def __init__(self, n_mod):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 5+1, 128)
        self.fc3 = nn.Linear(128 , n_mod)

    def forward(self, input1, constant):
        input1 = GradReverse.grad_reverse(input1, constant)
        logits = F.relu(self.fc1(input1))
        logits = self.fc3(logits)
        logits = F.log_softmax(logits, 1)

        return logits


