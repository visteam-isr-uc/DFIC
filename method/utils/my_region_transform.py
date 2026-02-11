#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:23:57 2022

@author: visteamer
"""


import math
import warnings
from collections.abc import Sequence
from typing import Tuple, List
import numbers
import torch
from torch import Tensor
import numpy as np
import random
import matplotlib.pyplot as plt


from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int



class RegionRandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), poi=None, 
                 scale_th=None, reg_coord=None, epoch = None, bb_max_dim = None, 
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        _log_api_usage_once(self)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.epoch = epoch
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.poi = poi
        self.filt_box = None
        self.scale_th = scale_th
        self.reg_coord = reg_coord
        self.bb_max_dim = bb_max_dim
            
            

    @staticmethod
    def get_params(img: Tensor, scale: List[float], scale_th: List[float], ratio: List[float], 
                   poi: List[Tuple[int, int]], reg_coord: List[List[Tuple[int, int]]], 
                   epoch: int, bb_max_dim: int) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
            scale_th (list): scale_th[0] is the probability of a scale random value to occur between scale[0] and scale_th[1]

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """

        _, height, width = F.get_dimensions(img)
        area = height * width
        if bb_max_dim is not None:
            min_dim = bb_max_dim
        else:
            min_dim=min(height,width)
        

        if epoch is not None:
            rng = random.Random(epoch)  
            # print(epoch)
        else:
            rand_state = random.getstate()
            rng = random.Random()
            rng.setstate(rand_state)
        

        log_ratio = torch.log(torch.tensor(ratio))
        mode_crop_scale = 0
        for _ in range(10):
            if scale_th:
                if random.uniform(0,1)<scale_th[0]:
                    # scale_rand = rng.uniform(scale[0],scale_th[1])
                    scale_rand = random.uniform(scale[0],scale_th[1])
                    target_area = (min_dim**2) * scale_rand
                    # print(scale[0], scale_th[1], scale_rand)
                else:
                    # mode_crop_scale = 1
                    scale_rand = random.uniform(scale_th[1],scale[1])
                    target_area = (min_dim**2) * scale_rand
                    # print(scale_th[1], scale[1], scale_rand)
                
            else:
                target_area = (min_dim**2) * random.uniform(scale[0],scale[1])
            aspect_ratio = torch.exp(random.uniform(log_ratio[0],log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # if 0 < w and 0 < h :
            if poi:
                if mode_crop_scale==1:
                    i, j, chosen_poi_index = get_i_j(w, h, width, height, [poi[-1]], epoch=epoch)
                else:
                    i, j, chosen_poi_index = get_i_j(w, h, width, height, poi, epoch=epoch)
            else:
                i = rng.randint(0, height - h) + np.random.normal(0.0, scale=5)
                j = rng.randint(0, width - w) + np.random.normal(0.0, scale=5)
                
                if i < 0:
                    i = 0
                elif i + h > height:
                    i = height-h
                    
                if j < 0:
                    j = 0
                elif j + w > width:
                    j = width - w
                
                # i = random.randint(0, height - h)
                # j = random.randint(0, width - w)
                


            if reg_coord:
                i, j, h, w = adjust_corners(i,j,h,w,height,width, poi, reg_coord, chosen_poi_index, img)
            return i, j, h, w
            


        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        i, j, h, w = self.get_params(img, self.scale, self.scale_th, self.ratio,
                                     self.poi, self.reg_coord, self.epoch, self.bb_max_dim)
        self.filt_box = (i, j, h, w)
        # if 3==3:
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        # else:
        #     n_ims = img.shape[0]
        #     container = torch.zeros((img.shape[0],img.shape[1],self.size[0],self.size[1]))
        #     for ind_im in range(img.shape[0]):
        #         container[ind_im] = F.resized_crop(img[ind_im], i, j, h, w, self.size, self.interpolation)
        #     return container

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string
    

def get_i_j(s_w, s_h, o_w, o_h, poi, poi_prob=None, epoch=None):
    
    
    if epoch is not None:
        n_rng = np.random.default_rng(epoch*2)
        # print('Ola')
        
        chosen_index = n_rng.choice(len(poi), p=poi_prob)
    #     chosen_poi = poi[chosen_index]
    #     i=int(round(chosen_poi[1]-s_h/2 + n_rng.normal(0.0, scale=5)))
    #     j=int(round(chosen_poi[0]-s_h/2 + n_rng.normal(0.0, scale=5)))
        
    else:
        
        chosen_index = np.random.choice(len(poi), p=poi_prob)
    
    scale = s_w*0.01
    chosen_poi = poi[chosen_index]
    i=int(round(chosen_poi[1]-s_h/2 + np.random.normal(0.0, scale=scale)))
    j=int(round(chosen_poi[0]-s_h/2 + np.random.normal(0.0, scale=scale)))

    
    # if i < 0:
    #     i = 0
    # elif i + s_h > o_h:
    #     i = o_h-s_h
        
    # if j < 0:
    #     j = 0
    # elif j + s_w > o_w:
    #     j = o_w - s_w
        
    return i, j, chosen_index

def find_conflits(i,j,h,w,reg_coord,chosen_poi_index):
    
    conflit_reg=[]
    present_reg=[chosen_poi_index]
    for ind, reg in enumerate(reg_coord):
        if ind != chosen_poi_index:
            pres_perc=0
            for p_x, p_y in reg:
                if j <= p_x < j+w and i <= p_y < i+h:
                    pres_perc += 1
            pres_perc = pres_perc/len(reg)
            if 0 < pres_perc < 0.5:
                conflit_reg.append((ind, pres_perc))
            elif pres_perc>=0.5:
                present_reg.append(ind)
                
    for it, (ind_c,_) in enumerate(conflit_reg):
        for ind_p in present_reg:
            if (ind_c==0 and ind_p==1) or (ind_c==1 and ind_p==0):
                # print(conflit_reg[it])
                del(conflit_reg[it])
                
            if (ind_c==2 and ind_p==3) or (ind_c==3 and ind_p==2):
                # print(conflit_reg[it])
                del(conflit_reg[it])
    return conflit_reg

def find_corner(c_p, r_p):

    l=c_p[0]<r_p[0]
    u=c_p[1]<r_p[1]
        
    if l and u:
        c=0
    elif l and not u:
        c=1
    elif not l and u:
        c=2
    else:
        c=3
    return c
            
def perform_resize_2adjust_increase(conflit_reg_i, i,j,h,w, height, width, c):
    conflit_reg=np.asarray(conflit_reg_i)
    margin=1
    if c==0:

        j_n,i_n=conflit_reg.min(axis=0)
        j_n = min(j,j_n-margin)
        i_n = min(i,i_n-margin)
            
        h_n=h + (i-i_n)
        w_n=w + (j-j_n)
        
        
    elif c==1:

        j_n=min(conflit_reg[:,0].min(axis=0)-margin, j)
        h_n=max(conflit_reg[:,1].max(axis=0)-i+margin, h)

        i_n=i
        w_n=w + (j-j_n)
    elif c==2:

        i_n=min(conflit_reg[:,1].min(axis=0)-margin, i)
        w_n=max(conflit_reg[:,0].max(axis=0)-j+margin, w)

        j_n=j
        h_n=h + (i-i_n)
        
    elif c==3:

        w_n=max(conflit_reg[:,0].max(axis=0)-j+margin, w)
        h_n=max(conflit_reg[:,1].max(axis=0)-i+margin, h)

        j_n=j
        i_n=i
        
    j_n=max(j_n,0)
    i_n=max(i_n,0)
    
    # w_n=min(j_n+w_n,width)
    # h_n=min(i_n+h_n,height)
    
    # print(c)
    # print(i,j,h,w)
    # print(i_n, j_n, h_n, w_n)
    return i_n, j_n, h_n, w_n

def perform_resize_2adjust_decrease(conflit_reg_i, i,j,h,w, height, width, c):
    conflit_reg=np.asarray(conflit_reg_i)
    or_rat = w/h
    maintain_rat = False
    margin=2
    if c==0:
        j_n,i_n=conflit_reg.max(axis=0)
        change_x = abs(j_n-j)
        change_y = abs(i_n-i)
        if change_x<change_y:
            j_f = max(j_n,j) + margin
            if maintain_rat:
                i_f = i + int((j_f-j)*(1/or_rat))
            else:
                i_f = i
        else:
            i_f = max(i_n, i) + margin
            if maintain_rat:
                j_f = j + int((i_f-i)*(or_rat))
            else:
                j_f = j
            
        h_f=h + (i-i_f)
        w_f=w + (j-j_f)
              
    elif c==1:

        j_n=conflit_reg[:,0].max(axis=0)
        h_n=conflit_reg[:,1].min(axis=0)-i
        change_x = abs(j_n-j)
        change_y = abs(h_n-h)
        if change_x<change_y:
            j_f = max(j_n,j) + margin
            if maintain_rat:
                h_f = h - int((j_f-j)*(1/or_rat))
            else:
                h_f = h
        else:
            h_f = min(h_n, h) - margin
            if maintain_rat:
                j_f = j + int((h-h_f)*(or_rat))
            else:
                j_f = j
            
        i_f = i
        w_f= w + (j-j_f)
    elif c==2:

        i_n=conflit_reg[:,1].max(axis=0)
        w_n=conflit_reg[:,0].min(axis=0)-j
        change_x = abs(w_n-w)
        change_y = abs(i_n-i)
        if change_x<change_y:
            w_f = min(w_n,w) - margin
            if maintain_rat:
                i_f = i + int((w-w_f)*(1/or_rat))
            else:
                i_f = i
        else:
            i_f = max(i_n, i) + margin
            if maintain_rat:
                w_f = w - int((i_f-i)*(or_rat))
            else:
                w_f = w
            
        j_f = j
        h_f= h + (i-i_f)
        
    elif c==3:

        h_n=conflit_reg[:,1].min(axis=0)-i
        w_n=conflit_reg[:,0].min(axis=0)-j
        change_x = abs(w_n-w)
        change_y = abs(h_n-h)
        if change_x<change_y:
            w_f = min(w_n,w) - margin
            if maintain_rat:
                h_f = i - int((w-w_f)*(1/or_rat))
            else:
                h_f = h
        else:
            h_f = min(h_n, h) - margin
            if maintain_rat:
                w_f = w - int((h-h_f)*(or_rat))
            else:
                w_f = w
            
        i_f = i
        j_f = j
        
    
    # w_n=min(j_n+w_n,width)
    # h_n=min(i_n+h_n,height)
    
    # print(c)
    # print(i,j,h,w)
    # print(i_n, j_n, h_n, w_n)
    return i_f, j_f, h_f, w_f

def plot_im(im):
    plt.imshow(im)
    plt.show()
    
def adjust_corners(i,j,h,w,height,width, poi, reg_coord, chosen_poi_index, img):
    conflit_reg=find_conflits(i,j,h,w,reg_coord,chosen_poi_index)
    # print(conflit_reg)
    # plot_im(img)
    # im0 = F.resized_crop(img, i, j, h, w, (128,128), InterpolationMode.BILINEAR)
    # plot_im(im0)
    while conflit_reg:
        reference_point=(j+w//2, i+h//2)
        counter=0
        for confl in conflit_reg:
            conflit_point=poi[confl[0]]
            conflit_reg_i = reg_coord[confl[0]]
            corner=find_corner(conflit_point, reference_point)
            increase=confl[1]>1
            # print(conflit_point)
            # print(reference_point)
            # print(corner)
            if increase:
                i,j,h,w= perform_resize_2adjust_increase(conflit_reg_i, i,j,h,w,height, width, corner)
            else:
                i,j,h,w= perform_resize_2adjust_decrease(conflit_reg_i, i,j,h,w,height, width, corner)
            counter+=1
            
            

            # im2 = F.resized_crop(img, i, j, h, w, (128,128), InterpolationMode.BILINEAR)
            # plot_im(im2)
        conflit_reg=find_conflits(i,j,h,w,reg_coord,chosen_poi_index)

    return i,j,h,w

        
        
    
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size