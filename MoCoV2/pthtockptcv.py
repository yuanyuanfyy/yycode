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
generate ckpt from pth.
(mindcv_resnet50; 线性分类pth转ckpt)
"""
import argparse
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pth_path', type=str,
                    help='pth path, eg:./data/example.pth')
parser.add_argument('--ckpt_path', type=str,
                    help='ckpt path, eg:./data/pretrain.ckpt')
args_opt = parser.parse_args()


if __name__ == '__main__':
    # pth2ckpt(args_opt.pth_path, args_opt.ckpt_path)
    params_dict1 = torch.load(args_opt.pth_path, map_location='cpu')
    #print("-------------------1--------------------", params_dict1)
    state_dict1 = params_dict1['state_dict']
    #state_dict1 = params_dict1
    #print("-------------------------2--------------------------", state_dict1)
    new_param_list = []
    for name in state_dict1:
        param_dict = {}
        parameter = state_dict1[name]
        if name.endswith('bn1.weight'):
            name = name[:name.rfind('bn1.weight')]
            name = name + 'bn1.gamma'
        elif name.endswith('bn1.bias'):
            name = name[:name.rfind('bn1.bias')]
            name = name + 'bn1.beta'
        elif name.endswith('bn1.running_mean'):
          name = name[:name.rfind('bn1.running_mean')]
          name = name + 'bn1.moving_mean'
        elif name.endswith('bn1.running_var'):
          name = name[:name.rfind('bn1.running_var')]
          name = name + 'bn1.moving_variance'
        elif name.endswith('bn2.weight'):
          name = name[:name.rfind('bn2.weight')]
          name = name + 'bn2.gamma'
        elif name.endswith('bn2.bias'):
          name = name[:name.rfind('bn2.bias')]
          name = name + 'bn2.beta'
        elif name.endswith('bn2.running_mean'):
          name = name[:name.rfind('bn2.running_mean')]
          name = name + 'bn2.moving_mean'
        elif name.endswith('bn2.running_var'):
          name = name[:name.rfind('bn2.running_var')]
          name = name + 'bn2.moving_variance'
        elif name.endswith('bn3.weight'):
          name = name[:name.rfind('bn3.weight')]
          name = name + 'bn3.gamma'
        elif name.endswith('bn3.bias'):
          name = name[:name.rfind('bn3.bias')]
          name = name + 'bn3.beta'
        elif name.endswith('bn3.running_mean'):
          name = name[:name.rfind('bn3.running_mean')]
          name = name + 'bn3.moving_mean'
        elif name.endswith('bn3.running_var'):
          name = name[:name.rfind('bn3.running_var')]
          name = name + 'bn3.moving_variance'
        elif name.endswith('downsample.0.weight'):
          name = name[:name.rfind('downsample.0.weight')]
          name = name + 'down_sample.0.weight'
        elif name.endswith('downsample.1.weight'):
          name = name[:name.rfind('downsample.1.weight')]
          name = name + 'down_sample.1.gamma'
        elif name.endswith('downsample.1.bias'):
          name = name[:name.rfind('downsample.1.bias')]
          name = name + 'down_sample.1.beta'
        elif name.endswith('downsample.1.running_mean'):
          name = name[:name.rfind('downsample.1.running_mean')]
          name = name + 'down_sample.1.moving_mean'
        elif name.endswith('downsample.1.running_var'):
          name = name[:name.rfind('downsample.1.running_var')]
          name = name + 'down_sample.1.moving_variance'
        elif name.endswith('fc.bias'):
          name = name[:name.rfind('fc.bias')]
          name = name + 'classifier.bias'
        elif name.endswith('fc.weight'):
          name = name[:name.rfind('fc.weight')]
          name = name + 'classifier.weight'
        elif name.endswith('num_batches_tracked'):
          continue
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        #param_dict['data'] = Tensor(parameter.detach().numpy())
        new_param_list.append(param_dict)
    save_checkpoint(new_param_list, args_opt.ckpt_path)
#python pthtockptcv.py --pth_path /old/fyy/mocov2/torch_premodel/moco_v1_200ep_pretrain.pth.tar --ckpt_path /old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_pretrain_mindcv.ckpt
#python pthtockptcv.py --pth_path /old/fyy/mocov2/torch_premodel/moco_v2_200ep_pretrain.pth.tar --ckpt_path /old/fyy/mocov2/mscode/pthtockpt_model/moco_v2_200ep_pretrain_mindcv.ckpt
#python pthtockptcv.py --pth_path /old/fyy/mocov2/torch_premodel/moco_v2_800ep_pretrain.pth.tar --ckpt_path /old/fyy/mocov2/mscode/pthtockpt_model/moco_v2_800ep_pretrain_mindcv.ckpt

#python pthtockptcv.py --pth_path /old/fyy/mocov2/torch_premodel/moco_v1_200ep_best.pth.tar --ckpt_path /old/fyy/mocov2/mscode/pthtockpt_model/moco_v1_200ep_best_mindcv.ckpt
#python pthtockptcv.py --pth_path /old/fyy/mocov2/torch_premodel/resnet50-19c8e357.pth --ckpt_path /old/fyy/mocov2/mscode/pthtockpt_model/resnet50-19c8e357_mindcv.ckpt
