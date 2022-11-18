"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import numpy as np

import mindspore.common.dtype as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.FastPose import createModel
from src.config import config


context.set_context(mode=context.GRAPH_MODE,
                    device_target=config.device_target,
                    device_id=config.device_id)

if __name__ == '__main__':
    net = createModel()
    # assert cfg.checkpoint_dir is not None, "cfg.checkpoint_dir is None."
    param_dict = load_checkpoint(config.ckpt_url)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.ones([1, 3, 256, 192]), ms.float32)
    export(net, input_arr, file_name=config.file_name,
           file_format=config.file_format)
