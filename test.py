from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm

# Specify the path to model config and checkpoint file
config_file = '/data/huminghe/mmdetection/configs/railway/gfl_0508_all_merged.py'
checkpoint_file = '/data/huminghe/mmdetection/model_0508_gfl/epoch_24.pth'
input_path = '/data/huminghe/test/detection_test'
output_path = '/data/huminghe/tmp/food_gfl'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

listdir = os.listdir(input_path)

for file_name in tqdm(listdir):
    if ".jpg" in file_name:
        img_path = os.path.join(input_path, file_name)
        result = inference_detector(model, img_path)
        out_file_path = os.path.join(output_path, file_name)

        model.show_result(img_path, result, out_file=out_file_path, visual_label_offset=5)
