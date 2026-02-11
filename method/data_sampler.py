
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import _pickle as cPickle
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


from PIL import Image
import numpy as np
import pandas as pd
import cv2
import random

from utils.my_region_transform import RegionRandomResizedCrop


pre_path = '../data/DFIC/preprocessed/'
masks_pre_path = '../data/DFIC/preprocessed/Masks_Torso/'


def surpress_labels(info_idx):
    
    label_weight = np.ones(26)
    
    if info_idx.eyes_closed in [1, 3] or info_idx.dark_tinted_lenses != 0 or info_idx.frame_too_heavy != 0 or info_idx.flash_reflection_lenses in [1,2] or info_idx.frame_covering_eyes != 0 or info_idx.too_dark_light == 2 or info_idx.hair_across_eyes != 0 or info_idx.roll_pitch_yaw in [1, 4, 5] or info_idx.non_neutral_expression != 0 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.other_faces_objects in ['b','c','d']:
        label_weight[0] = 0
        
    if info_idx.non_neutral_expression in [1, 5, 6, 7, 8] or (info_idx.non_neutral_expression == 0 and info_idx.mouth_open != 0) or info_idx.too_dark_light == 2 or info_idx.veil_over_face != 0 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.other_faces_objects in ['b','c','d']:
        label_weight[1] = 0
        
    if info_idx.mouth_open == 1 or info_idx.too_dark_light == 2 or info_idx.veil_over_face != 0 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.other_faces_objects in ['b','c','d']:
        label_weight[2] = 0
        
    if info_idx.rotated_shoulders == 1 or info_idx.too_dark_light == 2 or info_idx.other_faces_objects in ['b','c','d'] or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0:
        label_weight[3] = 0
        
    if info_idx.roll_pitch_yaw == 1 or info_idx.too_dark_light == 2 or info_idx.other_faces_objects in ['b','c','d'] or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0:
        label_weight[4] = 0
        
    if info_idx.looking_away == 1 or info_idx.eyes_closed in [1, 3] or info_idx.dark_tinted_lenses != 0 or info_idx.frame_covering_eyes != 0 or info_idx.frame_too_heavy != 0 or info_idx.flash_reflection_lenses in [1,2] or info_idx.too_dark_light == 2 or info_idx.hair_across_eyes != 0  or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.other_faces_objects in ['b','c','d']:
        label_weight[5] = 0
        
    if info_idx.hair_across_eyes == 1 or info_idx.too_dark_light == 2 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.other_faces_objects in ['b','c','d']:
        label_weight[6] = 0
        
    if info_idx.head_coverings == '1' or info_idx.too_dark_light != 0 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0:
        label_weight[7] = 0
        
    if info_idx.veil_over_face == 1 or info_idx.too_dark_light == 2 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0:
        label_weight[8] = 0
        
    if info_idx.other_faces_objects == '1' or info_idx.too_dark_light == 2 or info_idx.manipulated == True:
        label_weight[9] = 0
        
    if info_idx.dark_tinted_lenses == 1 or info_idx.too_dark_light == 2 or info_idx.manipulated == True:
        label_weight[10] = 0
        
    if info_idx.frame_covering_eyes == 1 or info_idx.too_dark_light == 2 or info_idx.manipulated == True:
        label_weight[11] = 0
        
    if info_idx.flash_reflection_lenses in [1, 3] or info_idx.too_dark_light == 2 or info_idx.manipulated == True:
        label_weight[12] = 0
        
    if info_idx.frame_too_heavy == 1 or info_idx.too_dark_light == 2 or info_idx.manipulated == True:
        label_weight[13] = 0
        
    if info_idx.shadows_behind_head in [1, 4] or info_idx.manipulated == True:
        label_weight[14] = 0
        
    if info_idx.shadows_across_face in [1, 4] or info_idx.head_coverings not in ['0', '8', 'a', 'c', 'e', 'g', 'i'] or info_idx.manipulated == True:
        label_weight[15] = 0
        
    if info_idx.flash_reflection_skin in [1, 4] or info_idx.manipulated == True:
        label_weight[16] = 0
        
    if info_idx.unnatural_skin_tone == 1 or info_idx.too_dark_light == 2 or info_idx.washed_out != 0 or info_idx.ink_marked_creased != 0:
        label_weight[17] = 0
        
    if info_idx.red_eyes == 1 or info_idx.dark_tinted_lenses != 0 or info_idx.frame_too_heavy != 0 or info_idx.flash_reflection_lenses in [1,2] or info_idx.frame_covering_eyes != 0 or info_idx.too_dark_light == 2 or info_idx.hair_across_eyes != 0 or info_idx.roll_pitch_yaw in [1, 4, 5, 8] or info_idx.non_neutral_expression != 0 or info_idx.ink_marked_creased != 0 or info_idx.blurred != 0 or info_idx.pixelation != 0 or info_idx.eyes_closed != 0:
        label_weight[18] = 0
        
    if info_idx.too_dark_light == 1:
        label_weight[19] = 0
        
    if info_idx.blurred in [1, 4] or info_idx.too_dark_light == 2:
        label_weight[20] = 0
        
    if info_idx.varied_background in [1, 8] or info_idx.shadows_behind_head != 0 or info_idx.ink_marked_creased != 0 or info_idx.posterization != 0:
        label_weight[21] = 0
        
    if info_idx.pixelation in [1, 4] or info_idx.too_dark_light == 2:
        label_weight[22] = 0
        
    if info_idx.washed_out in [1, 4] or info_idx.too_dark_light == 2:
        label_weight[23] = 0
        
    if info_idx.ink_marked_creased in [1, 4] or info_idx.too_dark_light == 2 or info_idx.varied_background != 0:
        label_weight[24] = 0
        
    if info_idx.posterization in [1, 4] or info_idx.too_dark_light == 2:
        label_weight[25] = 0
        
    return label_weight

class ICAODataset(Dataset):
    def __init__(self, transform=None, phase_train=True, phase_test=False, df_info = None, return_paths = False):
        
        
        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        
        if df_info is None:

            df_info_real = pd.read_csv('../data/DFIC/camera_image_labels.csv')
            df_info_real['im_path'] = df_info_real.im_path.apply(lambda x: 'Data_Torso/' + x)
            
            df_info_mani = pd.read_csv('../data/DFIC/artificial_image_labels.csv')
            df_info_mani['im_path'] = df_info_mani.im_path.apply(lambda x: 'Artificial_Torso/' + x)
            
            df_info = pd.concat([df_info_real, df_info_mani], ignore_index=True)
            
            if self.phase_train:
                ids_train = pd.read_csv('../data/DFIC/partitions/ids_all_train.txt', header = None)[0]
                df_info_train = df_info.loc[df_info.subj_id.isin(ids_train), :]
                
                final_info = df_info_train
                
            else:
                ids_test = pd.read_csv('../data/DFIC/partitions/ids_balanced_test.txt', header = None)[0]
                df_info_test = df_info.loc[df_info.subj_id.isin(ids_test), :]
                final_info = df_info_test
                
        else:
            final_info = df_info
            
        final_info.loc[:, 'other_faces_objects'] = final_info.loc[:, 'other_faces_objects'].astype('str')
            

        
        self.paths = final_info.loc[:, 'im_path'].to_list()
        self.complete_info = final_info
        self.manipulated = final_info.loc[:, 'manipulated'].to_list()
        self.return_paths = return_paths

        
        self.train_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512)])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):


        path = self.paths[idx]
        manipulated = self.manipulated[idx]
        
        data_dir, im_subj, im_dev, im_name = path.split('/')
        
        if manipulated:
            im_name = '_'.join(im_name.split('.')[0].split('_')[:-1]) + '.' + im_name.split('.')[-1]
        
        info_idx = self.complete_info.loc[self.complete_info.im_path == path, :].iloc[0]
        
        im = Image.open(pre_path + path)
        mask_path = masks_pre_path + im_subj + '/Camera/' + im_name.split('.')[0] + '_masks.pickle'
        
        im_pil_res = im.resize((512, 512), Image.Resampling.LANCZOS)

        with open(mask_path, "rb") as input_file:
            masks = cPickle.load(input_file)
            

        
        farl_masks = masks['farl_preds']
        eye_mask = masks['spiga_masks'][0]
        mouth_mask = masks['spiga_masks'][1]
        hc_mask = masks['ext_HC_mask']
        background_mask = ~masks['torso_mask']
        
        sunglasses_mask = farl_masks == 15
        hair_mask = farl_masks == 14


        face_mask = np.isin(farl_masks, [2,4,5,6,7,8,9,10,11,12,13])
        
        final_masks = np.stack([hc_mask, hair_mask, sunglasses_mask, eye_mask, mouth_mask, face_mask, ~background_mask, background_mask])
        final_masks = torch.tensor(final_masks)
        
        
        if self.phase_train:
            
            im = self.train_transforms(im)
            
            if random.uniform(0, 1) < 0.75: # 0.75
            
                third = 512//3
                middle = 512//2
                two_thirds = 1024//3
                
                pois = [(third,third), (third,middle), (third, two_thirds),
                        (middle,third), (middle,middle), (middle, two_thirds),
                        (two_thirds,third), (two_thirds,middle), (two_thirds, two_thirds)]
                
                i, j, h, w = RegionRandomResizedCrop.get_params(img = im_pil_res, scale = (0.6, 1.1), scale_th = None, ratio = (1,1),
                                         poi=pois, reg_coord = None, epoch = None, bb_max_dim=max(512, 512))
            
            
                im = F.resized_crop(im, i, j, h, w, 512, InterpolationMode.BILINEAR)
                final_masks = F.resized_crop(final_masks, i, j, h, w, 512, InterpolationMode.BILINEAR)
        
            
            if random.uniform(0, 1) < 0.5:
                im = F.hflip(im)
                final_masks = F.hflip(final_masks)
            
        
        if self.transform:
            im = self.transform(im)



        label_vec = np.array([info_idx.eyes_closed == 0,
                                info_idx.non_neutral_expression == 0,
                                info_idx.mouth_open == 0,
                                info_idx.rotated_shoulders == 0,
                                info_idx.roll_pitch_yaw == 0,
                                info_idx.looking_away == 0,
                                info_idx.hair_across_eyes == 0,
                                info_idx.head_coverings in ['0', '8', 'a', 'c', 'e', 'g', 'i'],
                                info_idx.veil_over_face == 0,
                                info_idx.other_faces_objects == '0',
                                info_idx.dark_tinted_lenses == 0,
                                info_idx.frame_covering_eyes == 0,
                                info_idx.flash_reflection_lenses == 0,
                                info_idx.frame_too_heavy == 0,
                                info_idx.shadows_behind_head == 0,
                                info_idx.shadows_across_face == 0,
                                info_idx.flash_reflection_skin == 0,
                                info_idx.unnatural_skin_tone == 0,
                                info_idx.red_eyes == 0,
                                info_idx.too_dark_light == 0,
                                info_idx.blurred == 0,
                                info_idx.varied_background == 0, 
                                info_idx.pixelation == 0,
                                info_idx.washed_out == 0,
                                info_idx.ink_marked_creased == 0,
                                info_idx.posterization == 0,]).astype('int')
        
        

        supervision_surpression = surpress_labels(info_idx)


        if self.return_paths:
            return im, final_masks, label_vec, supervision_surpression, path
        else:
            return im, final_masks, label_vec, supervision_surpression

if __name__ == '__main__':
    

    from utils.misc import deNormalize
    import matplotlib.pyplot as plt
    from utils.paint_image import fill_info
    
    
    
    label_map = np.array([
        (0, 128, 128),  # background
        (128, 0, 0),  # sunglasses
        (0, 128, 0),  # eyes
        (0, 0, 128),  # mouth
        (128, 0, 128),  # head coverings
        (128, 128, 0),  # hair coverings
    ])


    def draw_segmentation_map(mask, idex_color):
     

      
        red_map   = np.zeros_like(mask).astype(np.uint8)
        green_map = np.zeros_like(mask).astype(np.uint8)
        blue_map  = np.zeros_like(mask).astype(np.uint8)
      

        index = mask == True
         
        R, G, B = label_map[idex_color]
  
        red_map[index]   = R
        green_map[index] = G
        blue_map[index]  = B
      
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        return segmentation_map
    
    def image_overlay(image, segmented_image):
        alpha = 1  # transparency for the original image
        beta  = 0.9  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
      
        image = np.array(image)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         
        cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
        return image

    
    
    
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]
    
    
    normalize = transforms.Normalize(mean=means, 
                                      std=stds)
    

    val_dataset = ICAODataset( transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),phase_train=True,phase_test=False, return_paths = True)
    
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8,  num_workers=8,pin_memory=False, shuffle=True) # for large validation dataset, delete shuffle=True and introduce sampler=sampler_val
    
    counter = 0
    for i, (image, final_masks, label_vec, supervision_surpression, im_ident) in enumerate(val_loader):

        if counter%1000 == 0:
            print(counter)
        
        
        for im_ind in range(image.shape[0]):
            
            im_tens = image[im_ind]
            im_rgb = transforms.ToPILImage()(deNormalize(im_tens, means, stds)).convert('RGB')
            
            mask_hc = final_masks[im_ind, 0].numpy().astype('bool')
            mask_hair = final_masks[im_ind, 1].numpy().astype('bool')
            mask_sun = final_masks[im_ind, 2].numpy().astype('bool')
            mask_eye = final_masks[im_ind, 3].numpy().astype('bool')
            mask_mouth = final_masks[im_ind, 4].numpy().astype('bool')
            mask_face = final_masks[im_ind, 5].numpy().astype('bool')
            mask_back = final_masks[im_ind, 7].numpy().astype('bool')

            overlayed_image = image_overlay(im_rgb, draw_segmentation_map(mask_back,0))
            overlayed_image = image_overlay(overlayed_image, draw_segmentation_map(mask_hair,5))
            overlayed_image = image_overlay(overlayed_image, draw_segmentation_map(mask_sun,1))
            overlayed_image = image_overlay(overlayed_image, draw_segmentation_map(mask_eye,2))
            overlayed_image = image_overlay(overlayed_image, draw_segmentation_map(mask_mouth,3))
            overlayed_image = image_overlay(overlayed_image, draw_segmentation_map(mask_hc,4))
            
            
            overlayed_image_face = image_overlay(im_rgb, draw_segmentation_map(mask_face,1))
            overlayed_image_face = image_overlay(overlayed_image_face, draw_segmentation_map(~mask_back,2))
            
            label_ind = label_vec[im_ind]
            label_sup = supervision_surpression[im_ind]
            
            label_print = [int(x) for x in label_ind.detach().numpy()]
            sup_print = [int(x) for x in label_sup.detach().numpy()]
            
            plt.imshow(overlayed_image)
            plt.title(str(label_print) + '\n' + str(sup_print))
            plt.show()
            
            
            im_labs = fill_info(cv2.cvtColor(np.array(im_rgb), cv2.COLOR_RGB2BGR), label_print, sup_print)
            plt.imshow(cv2.cvtColor(im_labs, cv2.COLOR_BGR2RGB))
            plt.title(im_ident[im_ind])
            plt.show()
        counter += 1

        if counter >= 30:
            break  