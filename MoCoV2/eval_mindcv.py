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
##############test moco example on imagenet#################
python3 eval.py
"""
import numpy as np
import argparse
import ast
from mindspore import context, nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model

from src.dataset import create_dataset_lincls
from src.config import config
import mindcv



parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_id', type=int, default=5, help='device_id')
parser.add_argument('--distribute', type=ast.literal_eval, default=True)
parser.add_argument('--valdir', type=str, default='', help='ValData path')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='if mode is test, must provide path where the trained ckpt file')

args = parser.parse_args()

def test(ckpt_path):
    """run eval"""
    target = args.device_target
    # init context
    context.set_context(mode=context.GRAPH_MODE,
    #context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=target,
                        save_graphs=False)

    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    # dataset
    valdir = "/mass_store/dataset/imagenet/val"
    #valdir = "/old/fyy/mocov2/mscode/imagenet_test"
    val_loader = create_dataset_lincls(dataset_path=valdir, do_train=False,
                                        repeat_num=1, batch_size=config.batch_size,
                                        target=target, distribute=False)
    #for data in val_loader.create_dict_iterator():
    #    print('image', type(data["image"]), data["image"].shape, data["image"],
    #    'label', type(data["label"]), data["label"].shape, data["label"])
    step_size = val_loader.get_dataset_size()
    if step_size == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    # define net
    network = mindcv.create_model('resnet50')
    #network = mindcv.create_model('resnet50', num_classes=1001)
    #print(network)

    # load checkpoint
    ckpt_path = "/old/fyy/mocov2/mscode/result/mocov2-100_40037.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_best.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_best_mindcv.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/resnet50-19c8e357.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/resnet50-19c8e357_mindcv.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_pretrain.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_pretrain_mindcv.ckpt"
    #ckpt_path = "/old/fyy/mocov2/mscode/resnet50_ascend_v160_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt"
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    """
    param = network.parameters_dict()
    for par in param.keys():
      print("Parameter:\n", par, param[par].asnumpy())
    network.set_train(False)
    """
    #for data in val_loader.create_dict_iterator():
    #  output = network(data["image"])
    #  print('val_output', type(output), output.shape, output)
    # define loss, model
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    print("Dataset path: {}".format(args.valdir))
    print("Ckpt path :{}".format(ckpt_path))
    print("Class num: {}".format(1000))
    print("moco_lincls")
    print("============== Starting Testing ==============")
    acc = model.eval(val_loader)
    print("==============Acc: {} ==============".format(acc))


if __name__ == '__main__':
    path = args.ckpt_path
    test(path)
