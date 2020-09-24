import argparse
import numpy as np # linear algebra
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import os
from importlib import reload
import cvn_utils
import postprocess
from postprocess import PredictionPostProcessor
import math
from torch.utils.data.sampler import SubsetRandomSampler
from subprocess import Popen

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def to_tensor_transform():
    return transforms.Compose([           
        transforms.ToTensor(), 
    ])

def generate_image_tensor(image_fn):
    pil_img = Image.open(image_fn).convert("RGB")        
    return to_tensor_transform()(pil_img)

def find_all_jpeg_files(directory):
    assert os.path.exists(directory)
    fns = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if 'jpeg' not in filename:
                continue
            full_name = os.path.join(directory, filename)
            fns.append(full_name)
    return fns
    
#  python3 face_detection.py --image-directory /home/yangxu/face_mask_detection_workspace/data/cvn/vitamins-main --label-directory /home/yangxu/face_mask_detection_workspace/code/face_mask_detection/model_inference_labels --model-file model.pt

def main():
    parser = argparse.ArgumentParser(description='Face Detection using PyTorch')
    parser.add_argument('--image-directory', type=str, required=True, 
                        help='image directory')
    parser.add_argument('--label-directory', type=str, required=True,
                        help='label directory')
    parser.add_argument('--model-file', type=str, required=True,
                        help='model.pt file')
    parser.add_argument('--review', type=bool, required=False, default=False,
                        help='open labelme tool when finished')
    args = parser.parse_args()
    
    model = get_model_instance_segmentation(3)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print ('about to load model:', args.model_file)
    # load model
    model.load_state_dict(torch.load(args.model_file))
    print ('model loaded successfully!')
    model.eval()
    
    model.cuda()
    
    image_files = find_all_jpeg_files(args.image_directory)
    print('image file len = ', len(image_files))
    
    postprocessor = PredictionPostProcessor()
    
    for image_file in image_files:
        print('processing image file: ', image_file)
        postprocessor._image_path = image_file
        base = os.path.basename(image_file)
        base_name = os.path.splitext(base)[0]
        json_name = base_name +'.json' # + '.model_inference.json'
        json_file = os.path.join(args.label_directory, json_name)       
#         print('json file = ', json_file)
             
        image_tensor = generate_image_tensor(image_file)
#         print('image tensor:', image_tensor)
        input_tensors = [image_tensor.to(device)]
#         print('run model inference...')
        preds = model(input_tensors)
#         print('run model inference done!')
#         print('labels =', preds[0]['labels'])
        #assert preds is not None 
        assert len(preds) == 1
        postprocessor.convert_prediction_tensors_to_json_file(preds[0], json_file)
#         break
    if args.review:
        print('opening label tool for review...')
        Popen(['labelme', args.image_directory, '--output', args.label_directory])


if __name__ == '__main__':
    main()
