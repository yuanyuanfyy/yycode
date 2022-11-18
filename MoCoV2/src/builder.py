# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#import torch
#import torch.nn as nn
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.communication import init, get_rank, get_group_size
from mindspore import Parameter, Tensor
from mindspore.ops import stop_gradient, constexpr
from src.config import config
import mindspore.numpy as msnp

@constexpr
def generate_int(x):
    return x.asnumpy()

class MoCo(nn.Cell):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # class_num is the output fc dimension
        self.encoder_q = base_encoder(class_num=dim)
        self.encoder_k = base_encoder(class_num=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.SequentialCell([nn.Dense(dim_mlp, dim_mlp, weight_init='uniform', bias_init='uniform'), nn.ReLU(), self.encoder_q.fc])
            self.encoder_k.fc = nn.SequentialCell([nn.Dense(dim_mlp, dim_mlp, weight_init='uniform', bias_init='uniform'), nn.ReLU(), self.encoder_k.fc])

        for param_q, param_k in zip(self.encoder_q.get_parameters(), self.encoder_k.get_parameters()):
            #param_k._data = param_q.data.copy()  # initialize  #_data???可以吗
            param_k = param_q.copy()
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        shape = (dim, K)
        #self.queue = Parameter(ops.StandardNormal()(shape), name="queue", requires_grad = False)

        #l2_normalize_0 = ops.L2Normalize(axis=0, epsilon=1e-12)
        #self.queue.set_data(l2_normalize_0(self.queue))

        self.zeros = ops.Zeros()
        self.queue_ptr = Parameter(self.zeros(1, mstype.int32), name="queue_ptr", requires_grad = False)
        self.normalize = ops.L2Normalize(axis=1, epsilon=1e-12)
        self.matmul = ops.MatMul()
        self.expand_dims = ops.ExpandDims()
        self.concat_op = ops.Concat(axis=1)
        self.scatterupdate = ops.ScatterUpdate()
        self.zero = Tensor(0, mstype.int32)
        self.cast = ops.Cast()

    #@torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.get_parameters(), self.encoder_k.get_parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            #x = param_k.data * self.m + param_q.data * (1. - self.m)
            #param_k.set_data(x)


    #@torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = self.queue_ptr[0]
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        print("ptr=====", ptr)
        #print(batch_size.dtype)
        #self.queue_ptr[0] = ptr
        self.queue_ptr = self.queue_ptr.set_data(ptr)
        # = self.scatterndupdate(self.queue_ptr, self.zero, ptr)
        print("self.queue_ptr=====", self.queue_ptr)
        return 

    #@torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        #batch_size_this = x.shape[0]
        #print("x====", x.shape)
        #x_gather = concat_all_gather(x)
        #print("x_gather====", x_gather.shape)
        batch_size_all = x.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        randperm = ops.Randperm(dtype=mstype.int32) #ops.Randperm(max_length=1???)
        idx_shuffle = randperm(batch_size_all) #.cuda()???

        # broadcast to all gpus
        broadcast = ops.Broadcast(0)
        broadcast(idx_shuffle) #torch.distributed.broadcast广播到所有gpu

        # index for restoring
        sort = ops.Sort()
        _ ,idx_unshuffle = sort(idx_shuffle) #目前仅支持float16数据类型。如果使用float32类型可能导致数据精度损失。

        # shuffled index for this gpu
        init()
        gpu_idx = get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_this], idx_unshuffle

    #@torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        init()
        gpu_idx = get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def construct(self, im_q, im_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        #l2_normalize_1 = ops.L2Normalize(axis=1, epsilon=1e-12)
        q = self.normalize(q)

        # compute key features
        #with torch.no_grad():  # no gradient to keys
        #self._momentum_update_key_encoder()  # update the key encoder

        # shuffle for making use of BN
        #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        #l2_normalize_1 = ops.L2Normalize(axis=1, epsilon=1e-12)
        k = self.normalize(k)

        # undo shuffle
        #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        mmo = msnp.matmul(q, msnp.transpose(k, (1,0))).reshape(-1)
        axis = msnp.arange(0, 32*32, 33)
        l_posdim = msnp.take_along_axis(mmo, axis, 0)
        l_pos = self.expand_dims(l_posdim, -1)
        # negative logits: NxK
        l_neg = self.matmul(q, queue)  #????怎么改detach呢
        # logits: Nx(1+K)
        #concat_op = ops.Concat(axis=1)
        #cast_op = ops.Cast() #当前要求输入tensor的数据类型保持一致，若不一致时可通过ops.Cast把低精度tensor转成高精度类型再调用Concat算子。
        #logits = concat_op((cast_op(a, mindspore.float32), b))
        logits = self.concat_op((l_pos, l_neg)) #[l_pos, l_neg]
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = self.zeros(logits.shape[0], mstype.int32) #.cuda()
        # dequeue and enqueue
        #self._dequeue_and_enqueue(k)
        batch_size = k.shape[0]
        
        ptr = self.queue_ptr
        #assert self.K % batch_size == 0  # for simplicity
        #print("ptr=======", ptr.dtype)
        #ptr = self.cast(ptr, mstype.int32).asnumpy()
        # replace the keys at ptr (dequeue and enqueue)
        #self.queue[:, ptr:ptr + batch_size] = k.T
        self.queue_ptr = (ptr + batch_size) % self.K  # move pointer
        #ptr = self.expand_dims(ptr, 0)
        #print(batch_size.dtype)
        self.queue_ptr = ptr
        #self.queue_ptr.set_data(ptr)
        
        return logits, labels, k.T, ptr
        



# utils
#@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    #init()
    #group_size = get_group_size()
    #tensors_gather = [ops.ones_like(tensor)
    #    for _ in range(group_size)]
    #torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #print(tensor)
    allgather = ops.AllGather()
    tensors_gather = allgather(tensor) #???
    print(tensors_gather)
    #expand_dims = ops.ExpandDims()
    #tensors_gather = expand_dims(tensor, 0)

    concat_op = ops.Concat()
    #cast_op = ops.Cast() #当前要求输入tensor的数据类型保持一致，若不一致时可通过ops.Cast把低精度tensor转成高精度类型再调用Concat算子。
    #logits = concat_op((cast_op(a, mindspore.float32), b))
    output = concat_op(tensors_gather) #concat_op((tensors_gather))
    return output
    
class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn

    def construct(self, im_q, im_k, queue):
        logits, labels, kt, ptr = self.network(im_q, im_k, queue)
        loss = self._loss_fn(logits, labels)
        return loss, kt, ptr

   
class MoCoTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有两个：训练网络，优化器"""
        super(MoCoTrainOneStepCell, self).__init__(auto_prefix=False)
        self.sens = sens
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters     # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)  # 反向传播获取梯度
        self.grad_reducer = ops.Identity()
        self.cast = ops.Cast()

    def construct(self, im_q, im_k, queue):
        output = self.network(im_q, im_k, queue)                        # 计算当前输入的损失函数值
        loss, kt, ptr = output
        sens_tuple = (ops.ones_like(loss) * self.sens,)
        for i in range(1, len(output)):
            sens_tuple += (ops.zeros_like(output[i]),)
        grads = self.grad(self.network, self.weights)(im_q, im_k, queue, sens_tuple)  # 进行反向传播，计算梯度
        #grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss, kt, ptr



class MoCoEvalOneStepCell(nn.Cell):
    """自定义评估网络"""

    def __init__(self, network):
        super(MoCoEvalOneStepCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, im_q, im_k, queue):
        logits, labels, _, _ = self.network(im_q, im_k, queue)
        return logits, labels
