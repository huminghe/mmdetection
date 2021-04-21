from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm

# Specify the path to model config and checkpoint file
config_file = '/data/huminghe/mmdetection/configs/railway/yolov3_test.py'
checkpoint_file = '/data/huminghe/mmdetection/model_yolo_test/epoch_60.pth'
input_path = '/data/huminghe/data/doors/Images'
output_path = '/data/huminghe/test_result/door_yeva'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

listdir = os.listdir(input_path)

lists = list(listdir)[0: 300]

for file_name in tqdm(lists):
    if ".png" in file_name:
        img_path = os.path.join(input_path, file_name)
        result = inference_detector(model, img_path)
        out_file_path = os.path.join(output_path, file_name)

        model.show_result(img_path, result, out_file=out_file_path)
