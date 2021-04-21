from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm

# Specify the path to model config and checkpoint file
config_file = '/data/huminghe/mmdetection/configs/railway/gfl_0416_all_merged.py'
checkpoint_file = '/data/huminghe/mmdetection/model_gfl_0416_all_merged/epoch_24.pth'
input_path = '/data/huminghe/data/doors/Images'
output_path = '/data/huminghe/test_result/door_yeva'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

print(model.CLASSES[0])
print(model.CLASSES[1])
print(model.CLASSES[5])
