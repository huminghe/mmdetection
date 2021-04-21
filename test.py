from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm
import sys

config_file = sys.argv[1]
checkpoint_file = sys.argv[2]
input_path = sys.argv[3]
output_path = sys.argv[4]
device = int(sys.argv[5])

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:{}'.format(device))

listdir = os.listdir(input_path)

for file_name in tqdm(listdir):
    if ".jpg" in file_name:
        img_path = os.path.join(input_path, file_name)
        result = inference_detector(model, img_path)
        out_file_path = os.path.join(output_path, file_name)

        model.show_result(img_path, result, out_file=out_file_path)
