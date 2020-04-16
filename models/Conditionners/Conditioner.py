import torch.nn as nn

class Conditioner(nn.Module):
    def __init__(self):
        super(Conditioner, self).__init__()

    '''
    forward(self, x, context=None):
    :param x: A tensor [B, d]
    :param context: A tensor [B, c]
    :return: conditioning factors: [B, d, h] where h is the size of the embeddings.
    '''
    def forward(self, x, context=None):
        pass
