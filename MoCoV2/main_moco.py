# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Unsupervised Training 多卡
"""
import os
import argparse
import ast
from mindspore import context, nn, Tensor, ops
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore import dtype as mstype
import mindspore
from src.dataset import create_dataset_moco
from src.config import config
from src.resnet import resnet50
from src.builder import MoCo, MoCoTrainOneStepCell, WithLossCell, MoCoEvalOneStepCell

parser = argparse.ArgumentParser(description='Unsupervised Training')

#parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--distribute', type=ast.literal_eval, default=False)
parser.add_argument('--device_id', type=int, default=3, help='device_id')
parser.add_argument('--lr', default=0.03, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

args = parser.parse_args()


def main():
    target = args.device_target
    if args.distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, parameter_broadcast=True)
        init()
    else:
        # init context
        #context.set_context(mode=context.PYNATIVE_MODE,
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target,
                            save_graphs=False)
        if args.device_target == "Ascend":
            context.set_context(device_id=args.device_id)

    # dataset
    #traindir = os.path.join(args.data, 'train')
    traindir = "/mass_store/dataset/imagenet/train"

    train_datasetv2 = create_dataset_moco(dataset_path=traindir, aug_plus=True,
                                      repeat_num=1, batch_size=config.batch_size,
                                      target=target, distribute=args.distribute)
    train_datasetv1 = create_dataset_moco(dataset_path=traindir, aug_plus=False,
                                      repeat_num=1, batch_size=config.batch_size,
                                      target=target, distribute=False)

    #for data in train_datasetv2.create_dict_iterator():
    #    print('imageq', type(data["im_q"]), data["im_q"].shape, data["im_q"],
    #    'imagek', type(data["im_k"]), data["im_k"].shape, data["im_k"],
    #    'label', type(data["label"]), data["label"].shape, data["label"])

    #for data in train_datasetv1.create_dict_iterator():
        #print('imageq', type(data["im_q"]), data["im_q"].shape, data["im_q"],
        #'imagek', type(data["im_k"]), data["im_k"].shape, data["im_k"],
        #'label', type(data["label"]), data["label"].shape, data["label"])

    step_size_trainv2 = train_datasetv2.get_dataset_size()
    step_size_trainv1 = train_datasetv1.get_dataset_size()
    
    if step_size_trainv2 == 0:
        raise ValueError("Please check dataset_v2 size > 0 and batch_size <= dataset size")
    
    if step_size_trainv1 == 0:
        raise ValueError("Please check dataset_v1 size > 0 and batch_size <= dataset size")



    #net = resnet50(1001)
    model = MoCo(
        resnet50, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    #param_dict = mindspore.load_checkpoint('/old/fyy/mocov2/code/moco_v1_200ep_pretrain.ckpt'))
    #mindspore.load_param_into_net(model, param_dict)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_loss = WithLossCell(model, loss)
    optim = nn.SGD(model.trainable_params(), learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_net = MoCoTrainOneStepCell(net_with_loss, optim)
    train_net.set_train()
    #print(model)
    print("model", len(model.trainable_params()))
    eval_net = MoCoEvalOneStepCell(model)
    eval_net.set_train(False)
    topk1 = nn.Top1CategoricalAccuracy()
    topk5 = nn.Top5CategoricalAccuracy()
    topk1.clear()
    topk5.clear()

    shape = (128, 65536)
    queue = ops.StandardNormal()(shape)
    queue = ops.L2Normalize(axis=0, epsilon=1e-12)(queue)
    #ptr = 0
    for epoch in range(args.start_epoch, args.epochs):
        for data in train_datasetv2.create_dict_iterator():
            #output, target = model(im_q=data["im_q"], im_k=data["im_k"])
            #print('output', type(output), output.shape, output, 'target', type(target), target.shape, target)
            loss, kt, ptr = train_net(im_q=data["im_q"], im_k=data["im_k"], queue=queue)
            logits, labels = eval_net(im_q=data["im_q"], im_k=data["im_k"], queue=queue)
            print("logits, labels====", logits, labels)
            topk1.update(logits, labels)
            topk5.update(logits, labels)
            output_topk1 = topk1.eval()
            output_topk5 = topk5.eval()
            ptr = int(ptr)
            queue[:, ptr:ptr + config.batch_size] = kt
            print("loss====", loss)
            print("topk1, topk5====", topk1, topk5)
            print("output_topk1, output_topk5====", output_topk1, output_topk5)


    print("\n\n========================")
    #print("Dataset path: {}".format(args.data))
    print("Batch size: {}".format(config.batch_size))
    print("dataset_v2 size: {}".format(step_size_trainv2))
    #print("dataset_v1 size: {}".format(step_size_trainv1))
    print("=======Training begin========")


if __name__ == '__main__':
    main()
