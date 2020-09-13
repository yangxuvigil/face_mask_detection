import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import ntpath
import torch
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image

def resize_transform():
    return transforms.Compose([   
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def to_tensor_transform():
    return transforms.Compose([           
        transforms.ToTensor(), 
    ])
    
class JsonFileProcessor(object):
    new_width = 256.0
    new_height = 256.0
    def __init__(self, json_fn):
        self.json_fn = json_fn
        self.image_fn = None
        self.json_data = None
        self.width = None
        self.height = None
        self.original_boxes = []
        self.resized_boxes = []
        self.labels = []
        self.loaded = False
        
    def load(self):
        self.load_json_data()
        self.load_image_fn()
        self.load_boxes_and_labels()
        self.loaded = True
        
    def load_image_fn(self):        
        self.image_fn = self.find_image_file()
    
    def load_json_data(self):
        with open(self.json_fn) as json_file:
            self.json_data = json.load(json_file)
            self.width = self.json_data['imageWidth']
            self.height = self.json_data['imageHeight']   
    
    def compute_new_x(self, x):
        return (x / self.width) * self.new_width
    
    def compute_new_y(self, y):
        return (y / self.height) * self.new_height
        
    def compute_original_box(self, shape):
        xmin = shape['points'][0][0]
        ymin = shape['points'][0][1]
        xmax = shape['points'][1][0]
        ymax = shape['points'][1][1]
        if ymin > ymax:            
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        assert (xmin <= xmax)
        assert (ymin <= ymax)
        return [xmin, ymin, xmax, ymax]
    
    def compute_resized_box(self, original_box):
        xmin, ymin, xmax, ymax = original_box
        return [self.compute_new_x(xmin), 
                self.compute_new_y(ymin),
                self.compute_new_x(xmax),
                self.compute_new_y(ymax)]
    
    def compute_label(self, shape):
        if shape['flags']['mask']:
            return 1
        else:
            return 0
            
    def load_boxes_and_labels(self):        
        for shape in self.json_data['shapes']:                 
            original_box = self.compute_original_box(shape)
            self.original_boxes.append(original_box)
            self.resized_boxes.append(self.compute_resized_box(original_box))
            self.labels.append(self.compute_label(shape))        
    
    def find_image_file(self):
        assert os.path.exists(self.json_fn)
        head, tail = ntpath.split(self.json_fn)

        head, tail = ntpath.split(head)

        with open(self.json_fn) as json_file:
            data = json.load(json_file)
            image_path = data['imagePath']
            image_file = image_path.split('/')[1]
            image_full_name = os.path.join(head, image_file)
            image_full_name = image_full_name.replace(':', '_')
            assert os.path.exists(image_full_name)
            return image_full_name             
        
    def generate_target(self, idx):
        boxes_tensor = torch.as_tensor(self.original_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(self.labels, dtype=torch.int64)
        img_id_tensor = torch.tensor([idx])
        target = {}
        target['boxes'] = boxes_tensor
        target['labels'] = labels_tensor
        target['image_id'] = img_id_tensor
        return target
    
    def generate_resized_image_tensor(self):
        pil_img = Image.open(self.image_fn).convert("RGB")        
        return resize_transform()(pil_img)
    
    def generate_image_tensor(self):
        pil_img = Image.open(self.image_fn).convert("RGB")        
        return to_tensor_transform()(pil_img)
        
    def plot_original_image(self):
        if not self.loaded:
            print('The processor has been loaded!')
            return
        pil_img = Image.open(self.image_fn).convert("RGB")
        w, h = pil_img.size
        img_tensor = to_tensor_transform()(pil_img)

        fig,ax = plt.subplots(1)
        img = img_tensor.cpu().data

        # Display the image
        ax.imshow(img.permute(1, 2, 0))
        for box in self.original_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()        
        
    def plot_resized_image(self):
        if not self.loaded:
            print('The processor has been loaded!')
            return
        pil_img = Image.open(self.image_fn).convert("RGB")
        w, h = pil_img.size
        img_tensor = resize_transform()(pil_img)

        fig,ax = plt.subplots(1)
        img = img_tensor.cpu().data

        # Display the image
        ax.imshow(img.permute(1, 2, 0))
        for box in self.resized_boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()