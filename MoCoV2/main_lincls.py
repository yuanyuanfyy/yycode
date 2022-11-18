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
Linear Classification
"""
import os
import argparse
import ast
import mindspore
from mindspore import context, ops, Tensor, nn, Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.resnet import resnet50
from src.dataset import create_dataset_lincls
from src.config import config
from src.lr_scheduler import get_lr

parser = argparse.ArgumentParser(description='Linear Classification')

#parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--distribute', type=ast.literal_eval, default=False)
parser.add_argument('--device_id', type=int, default=5, help='device_id')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--pretrained', type=str, default="/old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_pretrain.ckpt", help='path to moco pretrained checkpoint')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--lr', default=25., type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=0., type=float,
                    help='weight decay (default: 0.)')
     
args = parser.parse_args()


def main():
    target = args.device_target
    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target,
                        save_graphs=False)

    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    # dataset
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    traindir = "/mass_store/dataset/imagenet/train"
    valdir = "/mass_store/dataset/imagenet/val"
    

    train_loader = create_dataset_lincls(dataset_path=traindir, do_train=True,
                                        repeat_num=1, batch_size=config.batch_size,
                                        target=target, distribute=False)
    val_loader = create_dataset_lincls(dataset_path=valdir, do_train=False,
                                        repeat_num=1, batch_size=config.batch_size,
                                        target=target, distribute=False)
    #for data in train_loader.create_dict_iterator():
    #    print('image', type(data["image"]), data["image"].shape, data["image"],
    #    'label', type(data["label"]), data["label"].shape, data["label"])
    '''
    for data in val_loader.create_dict_iterator():
        print('image', type(data["image"]), data["image"].shape, data["image"],
        'label', type(data["label"]), data["label"].shape, data["label"])
    '''
    step_size_train = train_loader.get_dataset_size()
    step_size_val = val_loader.get_dataset_size()
    if step_size_train == 0:
        raise ValueError("Please check dataset_train size > 0 and batch_size <= dataset size")
    
    if step_size_val == 0:
        raise ValueError("Please check dataset_val size > 0 and batch_size <= dataset size")
    
    
    lr = get_lr(lr_init=args.lr,
                schedule=args.schedule,
                total_epochs=args.epochs,
                steps_per_epoch=step_size_train)

    lr = Tensor(lr)
    
    network = resnet50(1000)
    for param in network.trainable_params():
        if param.name not in ['end_point.weight', 'end_point.bias']:
            param.requires_grad = False
            
    network.end_point.weight.set_data(ops.normal((1000, 2048), mean=Tensor(0.0), stddev=Tensor(0.01)))
    network.end_point.bias.set_data(ops.Zeros()((1000,), mindspore.float32))
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = mindspore.load_checkpoint(args.pretrained)

            # rename moco pre-trained keys
            for k in list(checkpoint.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.end_point'):
                    # remove prefix
                    checkpoint[k[len("module.encoder_q."):]] = checkpoint[k]
                # delete renamed or unused k
                del checkpoint[k]
    
            args.start_epoch = 0
            msg = mindspore.load_param_into_net(network, checkpoint)
            assert set(msg) == {"end_point.weight", "end_point.bias"}
    
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    #print(network.trainable_params())
    optim = nn.SGD(network.trainable_params(), learning_rate=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = Model(network, loss_fn=loss, optimizer=optim, metrics={'acc'})
    
    time_cb = TimeMonitor(data_size=step_size_train)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size_train,
                                     keep_checkpoint_max=10)

        if args.isModelArts:
            save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
        else:
            if target == "GPU" and args.distribute:
                save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(get_rank()) + '/')
            else:
                save_checkpoint_path = config.save_checkpoint_path

        ckpt_cb = ModelCheckpoint(prefix="mocov2",
                                  directory=save_checkpoint_path,
                                  config=config_ck)
        cb += [ckpt_cb]
    print("=======Training begin========")    
    model.train(args.epochs, train_loader, callbacks=cb, dataset_sink_mode=True)
        
    #####学习率事宜
    print("\n\n========================")
    #print("Dataset path: {}".format(args.data))
    print("Batch size: {}".format(config.batch_size))
    print("dataset_train size: {}".format(step_size_train))
    print("dataset_val size: {}".format(step_size_val))



if __name__ == '__main__':
    main()
