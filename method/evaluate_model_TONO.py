

import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import numpy as np


from data_sampler_TONO import TONO_Dataset
from models.architecture.DeepLabV3_ICAO_SE_lite import ICAO_DEEPLAB
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from tabulate import tabulate
import _pickle as cPickle


model_path = 'models/TONO_best.pth.tar'  




model = ICAO_DEEPLAB(n_maps = 8, n_reqs = 26)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only = False)
model.load_state_dict(checkpoint['state_dict'])

model.eval()


# list images

dirs_path = '../data/TONO/Data/'
labels = [lab for lab in os.listdir(dirs_path) if os.path.isdir(dirs_path + lab)]
lab_list = []
im_path_list = []
for lab in labels:
    ims = os.listdir(dirs_path + lab)
    
    for im in ims:
    
        lab_list.append(lab)
        im_path_list.append(dirs_path + lab + '/' + im)
        
df_tono = pd.DataFrame({'im_path': im_path_list, 'label': lab_list})
    

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]


normalize = transforms.Normalize(mean=means, 
                                  std=stds)


val_dataset = TONO_Dataset( transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    normalize
]), df_info = df_tono)


val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32,  num_workers=8,pin_memory=False, shuffle=False) 

counter = 0
sigmoid_fun = torch.nn.Sigmoid()
yv, xv = np.meshgrid(list(range(512)), list(range(512)), indexing='ij')

labels = []
predictions = []
im_paths = []

model.cuda()
for i, (image, labels_by_reg, paths) in enumerate(val_loader):
    
    
    with torch.no_grad():
        parser_masks, infered_classes_by_reg = model(image.cuda())

    if counter%10 == 0:
        print(counter)
        
        
    batch_preds = 1 - sigmoid_fun(infered_classes_by_reg).to('cpu').numpy()
        

    
    # Common
    predictions.append(batch_preds)
    labels.extend(list(labels_by_reg))
    im_paths.extend(list(paths))

    counter += 1

predictions_np = np.concatenate(predictions)


df_res = pd.DataFrame({'im_path': im_paths, 'label': labels})



names = ['eyes_closed',
             'non_neutral_expression',
             'mouth_open',
             'rotated_shoulders',
             'roll_pitch_yaw',
             'looking_away',
             'hair_across_eyes',
             'head_coverings',
             'veil_over_face',
             'other_faces_objects',
             'dark_tinted_lenses',
             'frame_covering_eyes',
             'flash_reflection_lenses',
             'frame_too_heavy',
             'shadows_behind_head',
             'shadows_across_face',
             'flash_reflection_skin',
             'unnatural_skin_tone',
             'red_eyes',
             'too_dark_light',
             'blurred',
             'varied_background',
             'pixelation',
             'washed_out',
             'ink_marked_creased',
             'posterization']
dict_scos = {names[i]: list(predictions_np[:,i]) for i in range(len(names))} 

df_res = pd.concat([df_res, pd.DataFrame(dict_scos)], axis = 1)



tono_label_model_trigger_map = {
    'bkg': {'bkg': ['varied_background']}, 
    'cap': {'cap': ['head_coverings']},
    'ceg': {'ceg': ['eyes_closed']},
    'expos': {'expos': ['too_dark_light']},
    'la': {'la_1': ['combined_gaze_metric'], 'la_2': ['combined_gaze_metric']}, 
    'light': {'light': ['too_dark_light', 'shadows_across_face',]}, 
    'oof': {'oof': ['blurred']},
    'pixel': {'pixel': ['pixelation']},
    'poster': {'poster': ['posterization']},
    'sat': {'sat': ['unnatural_skin_tone']}, 
    'sm': {'sm': ['non_neutral_expression', 'mouth_open']},
    'sun': {'sun': ['dark_tinted_lenses']},
    'tq': {'tq': ['rotated_shoulders', 'roll_pitch_yaw']}} 


def get_eer(labels, predictions):

   
    fpr, tpr, ths = roc_curve(labels, predictions, drop_intermediate = False)
    
    interp_func = interp1d(fpr, tpr)

    fpr_interp = np.arange(0, 1, 0.001)
    tpr_interp = interp_func(fpr_interp)

    
    znew = abs(fpr_interp + tpr_interp -1)
    eer = 1 - tpr_interp[np.argmin(znew)]
    best_hter = np.min((fpr + 1 - tpr)/2)
    auc = roc_auc_score(labels, predictions)
    

    

 
    return float(auc), float(eer), float(best_hter)


##### Get Spiga Gaze
left_eye_indexes = list(range(60,68))
right_eye_indexes = list(range(68,76))




folders = [x for x in os.listdir(dirs_path) if os.path.isdir(dirs_path+x)]

gaze_metric = []
im_path = []
label = []

for tono_dir in folders:
    
    ann_files = os.listdir(dirs_path + tono_dir)
    
    for ann in ann_files:
        
        lands_path_corr = '../data/TONO/LandMarks/' + tono_dir + '/' + ann.split('.')[0] + '_spiga.pickle'
        with open(lands_path_corr, "rb") as input_file:
            lands = cPickle.load(input_file)[0]['landmarks']
            
            
        left_eye_lands = lands[left_eye_indexes]
        right_eye_lands = lands[right_eye_indexes]
        
        pupil_left_center_roi = lands[96]
        pupil_right_center_roi = lands[97]
        
        left_norm_factor = max(left_eye_lands[:,0].max() - pupil_left_center_roi[0], 
                               pupil_left_center_roi[0] - left_eye_lands[:,0].min())
        
        right_norm_factor = max(right_eye_lands[:,0].max() - pupil_right_center_roi[0], 
                               pupil_right_center_roi[0] - right_eye_lands[:,0].min())
        
        para = 1
        
        left_mean_x = (left_eye_lands[:,0].max() + left_eye_lands[:,0].min())/2
        right_mean_x = (right_eye_lands[:,0].max() + right_eye_lands[:,0].min())/2
        
        left_x_dist = 1 - abs(left_mean_x - pupil_left_center_roi[0])/left_norm_factor
        right_x_dist = 1 - abs(right_mean_x - pupil_right_center_roi[0])/right_norm_factor
        
        mean_dist = ((left_mean_x - pupil_left_center_roi[0])/left_norm_factor + \
            (right_mean_x - pupil_right_center_roi[0])/right_norm_factor)/2
            
        gaze_metric.append(1-abs(mean_dist))
        
        im_path.append(dirs_path + tono_dir + '/' + ann)
        label.append(tono_dir)
        
        
df = pd.DataFrame({'im_path': im_path, 'label': label, 'gaze_metric': gaze_metric})


# #####

df_res = pd.merge(df_res, df.loc[:,['im_path', 'gaze_metric']], how='left', on='im_path', suffixes=('', '_y'))
df_res['combined_gaze_metric'] = df_res.apply(lambda x: x.gaze_metric if (x.looking_away > 0.5 and x.roll_pitch_yaw > 0.5) else x.looking_away, axis = 1)

AUCs = []
EERs = []
HTERs = []

indexes = []
for tono_lab, dict_dir_reqs in tono_label_model_trigger_map.items():
    
    
    all_preds_uncmp = []

    for dir_lab, model_reqs in dict_dir_reqs.items():
    
        preds_uncmp = df_res.loc[df_res.label == dir_lab, model_reqs].to_numpy().mean(axis = 1)

        all_preds_uncmp.append(preds_uncmp)

        
    all_preds_uncmp_np = np.concatenate(all_preds_uncmp)

    preds_cmp = df_res.loc[df_res.label == 'icao', model_reqs].to_numpy().mean(axis = 1)
    
    labs_cmp = np.ones(len(preds_cmp)).astype('int')
    labs_uncmp = np.zeros(len(all_preds_uncmp_np)).astype('int')
    
    all_preds = np.concatenate([preds_cmp, all_preds_uncmp_np])
    all_labs = np.concatenate([labs_cmp, labs_uncmp])
    

    
    auc, eer, hter = get_eer(all_labs, all_preds)
    

    AUCs.append(auc)
    EERs.append(eer)
    HTERs.append(hter)
    

    indexes.append(tono_lab)
    
    para = 1
    
indexes.append('AVG')
AUCs.append(np.array(AUCs).mean())
EERs.append(np.array(EERs).mean())
HTERs.append(np.array(HTERs).mean())


    
df_final = pd.DataFrame(data = {'AUC': AUCs, 'EER': EERs, 'HTER': HTERs}, index = indexes)
    
table_form = tabulate(df_final, headers = 'keys', tablefmt = 'fancy_grid')
print(table_form)


