

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

from method.models.architecture.DeepLabV3_ICAO_SE_lite import ICAO_DEEPLAB


class ICAO_Model:
    
    def __init__(self, model_path = 'method/models/DFIC_best.pth.tar'):
        
        model = ICAO_DEEPLAB(n_maps = 8, n_reqs = 26)
        model_path = model_path
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only = False)
        model.load_state_dict(checkpoint['state_dict'])

        self.model = model.eval()
        
        
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=means, 
                                          std=stds)

        trans4model = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            normalize
        ])
        
        self.pre_trans = trans4model
        
        
    def process(self, im_cv_bgr):
        
        sigmoid_fun = torch.nn.Sigmoid()
        
        im_cv_rgb = cv2.cvtColor(im_cv_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_cv_rgb)
        
        im4model = self.pre_trans(im_pil).unsqueeze(0)
        
        with torch.no_grad():
            infered_masks, infered_classes_by_reg = self.model(im4model)
            
        final_masks = sigmoid_fun(infered_masks).detach().numpy()[0] > 0.75
        final_preds = 1 - sigmoid_fun(infered_classes_by_reg).numpy()[0]
        
        return final_masks, final_preds
    

