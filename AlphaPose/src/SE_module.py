'''
Alphapose network
'''
import mindspore.nn as nn
import mindspore.ops as ops
class SELayer(nn.Cell):
    '''
    SELayer
    '''
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell(
            [nn.Dense(channel, channel // reduction),
             nn.ReLU(),
             nn.Dense(channel // reduction, channel),
             nn.Sigmoid()]
        )

    def construct(self, x):
        '''
        construct
        '''
        b, c, _, _ = x.shape
        y = self.avg_pool(x, (2, 3)).view((b, c))
        y = self.fc(y).view((b, c, 1, 1))
        return x * y
