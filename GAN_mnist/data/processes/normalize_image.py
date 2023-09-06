import numpy as np
import torch
import warnings
from concern.config import State

from .data_process import DataProcess
from torchvision import transforms

class NormalizeImage(DataProcess):
    RGB_MEAN = np.array(0.5)
    norm_type = State(default="lib")
 
    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        self.debug = False
        warnings.simplefilter("ignore")
                
        self.inv_normal = transforms.Compose([ 
           transforms.Normalize(mean=(0), std=(1/0.5)), 
           transforms.Normalize(mean=(-0.5), std=(1)), 
           ])
        self.cnormalize = self.build_transform()

    def build_transform(self):

        transform = transforms.Compose(
            [
               # transforms.ToPILImage(),
                transforms.ToTensor(),    # HWC -> CHW
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )
        return transform        
            
    def lib_trans(self, image):
        
        
        return self.cnormalize(image)
        #return self.normalize(torch.from_numpy(image).permute(2, 1, 0)).float()
        
    def lib_inv_trans(self, image):
        
        image1=(self.inv_normal(image.to('cpu')).permute(1,2,0).numpy()*255).astype('uint8') 
        
        
        return np.ascontiguousarray(image1)
        
    def manual_trans(self, image):
        
        image -= self.RGB_MEAN
        image /= 255.
        #image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.from_numpy(image).permute(2, 1, 0).float()
        
        return image
    
    def manual_inv_trans(self, image):
        #image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image.permute(2, 1, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)       
        
        return image

    def process(self, data):
        
        assert 'image' in data, '`image` in data is required by this process'
        image = data['image']
        
        if(self.norm_type == 'lib'):
            
            data['image'] = self.lib_trans(image.astype('uint8'))
        else:
            data['image'] = self.manual_trans(image.astype('uint8'))
        
        return data
        #assert 'image' in data, '`image` in data is required by this process'
        #image = data['image']
        #image -= self.RGB_MEAN
        #image /= 255.
        #image = torch.from_numpy(image).permute(2, 0, 1).float()
        #image = torch.from_numpy(image).permute(2, 1, 0).float()
        #data['image'] = image
        #return data

    @classmethod
    def restore(self, image):
        #image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image.permute(2, 1, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image
