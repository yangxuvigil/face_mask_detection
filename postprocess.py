import numpy as np # linear algebra
import json
import ntpath
import os
import torch
import torchvision
from torchvision import transforms, datasets, models


def convert_one_box_label_to_shape_dict(box, label, index):
    xmin, ymin, xmax, ymax = box
    has_mask = True if label in [1, 2] else False    
    shape_dict = {'label' : 'face',
                  'points' : [
                      [xmin, ymin],
                      [xmax, ymax]
                  ],
                  'group_id': index,
                  'shape_type' : 'rectangle',
                  'flags' : {
                      'mask': has_mask
                  },
                  'occluded' : False
                  }
    return shape_dict

def convert_prediction_tensors_to_shapes(prediction):    
    labels = prediction['labels'].detach().cpu().numpy().tolist()
    boxes = prediction['boxes'].detach().cpu().numpy().tolist()
    assert len(labels) == len(boxes)
    shapes = []    
    
    for idx in range(len(labels)):
        shapes.append(convert_one_box_label_to_shape_dict(boxes[idx], labels[idx], idx))
    return shapes

class PredictionPostProcessor(object):
    
    def __init__(self):
        self._version = '4.5.6'
        self._flags = {}
        self._image_path = 'unknown'
        self._image_data = None
        self._image_height = None
        self._image_width = None
    
    def convert_prediction_tensors_to_json_file(self, prediction, json_file):
        dict = {'version' : self._version,
                'flags' : self._flags,
                'imagePath' : self._image_path,
                'imageData' : self._image_data,
                #'imageHeight': self._iamge_height
                #'imageWidth': self._image_width,
                'shapes' : convert_prediction_tensors_to_shapes(prediction)}
        with open(json_file, 'w') as out_file:
            json.dump(dict, out_file, indent=4)
    
