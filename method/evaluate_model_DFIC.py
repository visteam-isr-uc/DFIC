
import pandas as pd
import torch
import torchvision.transforms as transforms
import numpy as np

from data_sampler_4test import ICAODataset
from models.architecture.DeepLabV3_ICAO_SE_lite import ICAO_DEEPLAB
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from tabulate import tabulate




# model_path = 'models/TONO_best.pth.tar'  
model_path = 'models/DFIC_best.pth.tar'  


df_info_real = pd.read_csv('../data/DFIC/camera_image_labels.csv')
df_info_real['im_path'] = df_info_real.im_path.apply(lambda x: 'Data_Torso/' + x)

df_info_mani = pd.read_csv('../data/DFIC/artificial_image_labels.csv')
df_info_mani['im_path'] = df_info_mani.im_path.apply(lambda x: 'Artificial_Torso/' + x)

df_info = pd.concat([df_info_real, df_info_mani], ignore_index=True)

ids_test = pd.read_csv('../data/DFIC/partitions/ids_balanced_test.txt', header=None)[0]
df_info_test = df_info.loc[df_info.subj_id.isin(ids_test), :]



model = ICAO_DEEPLAB(n_maps = 8, n_reqs = 26)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only = False)
model.load_state_dict(checkpoint['state_dict'])

model.eval()



means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]


normalize = transforms.Normalize(mean=means, 
                                  std=stds)


val_dataset = ICAODataset( transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    normalize
]), df_info = df_info_test)


val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32,  num_workers=8,pin_memory=False, shuffle=False) # for large validation dataset, delete shuffle=True and introduce sampler=sampler_val

counter = 0
sigmoid_fun = torch.nn.Sigmoid()


labels = []
predictions = []
im_paths = []

model.cuda()
for i, (image, paths) in enumerate(val_loader):
    
    
    with torch.no_grad():
        parser_masks, infered_classes_by_reg = model(image.cuda())

    if counter%10 == 0:
        print(counter)
        
        
    batch_preds = 1 - sigmoid_fun(infered_classes_by_reg).to('cpu').numpy()
        

    
    # Common
    predictions.append(batch_preds)
    im_paths.extend(list(paths))

    counter += 1

predictions_np = np.concatenate(predictions)


df_res = pd.DataFrame({'im_path': im_paths})

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
    

    
df_scos_test = df_res.copy()
df_demo = pd.read_csv('../data/DFIC/demographic_info.csv')

age_cat = '*' # Children/Teen ; Young Adult ; Adult ; Senior Adult ; Senior ; *
gender = '*' # Male ; Female ; *
ethnicity = '*' # Caucasian ; African ; Asian ; *

filtered_df = df_demo.copy()
if age_cat != '*':
    filtered_df = filtered_df.loc[filtered_df.age_cat == age_cat, :]
if gender != '*':
    filtered_df = filtered_df.loc[filtered_df.gender == gender, :]
if ethnicity != '*':
    filtered_df = filtered_df.loc[filtered_df.ethnicity == ethnicity, :]
    
test_filt_subjs = df_info_test.loc[df_info_test.subj_id.isin(filtered_df.subj_id), 'subj_id'].unique()

facing_labs_model_reqs_map = {
    'eyes_closed': ['eyes_closed'], #, 'shadows_behind_head', 'other_faces_objects', 'ink_marked_creased'
    'non_neutral_expression': ['non_neutral_expression'],
    'mouth_open': ['mouth_open'],
    'rotated_shoulders': ['rotated_shoulders'],
    'roll_pitch_yaw': ['roll_pitch_yaw'],
    'looking_away': ['looking_away'],
    'hair_across_eyes': ['hair_across_eyes'],
    'head_coverings': ['head_coverings'],
    'veil_over_face': ['veil_over_face'],
    'other_faces_objects': ['other_faces_objects'],
    'dark_tinted_lenses': ['dark_tinted_lenses'],
    'frame_covering_eyes': ['frame_covering_eyes'],
    'flash_reflection_lenses': ['flash_reflection_lenses'],
    'frame_too_heavy': ['frame_too_heavy'],
    'shadows_behind_head': ['shadows_behind_head'],
    'shadows_across_face': ['shadows_across_face'],
    'flash_reflection_skin': ['flash_reflection_skin'],
    'unnatural_skin_tone': ['unnatural_skin_tone'],
    'red_eyes': ['red_eyes'],
    'too_dark_light': ['too_dark_light'],
    'blurred': ['blurred'],
    'varied_background': ['varied_background'],
    'pixelation': ['pixelation'],
    'washed_out': ['washed_out'],
    'ink_marked_creased': ['ink_marked_creased'],
    'posterization': ['posterization']}
 


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



AUCs = []
EERs = []
HTERs = []


indexes = []

for facing_lab, model_reqs in facing_labs_model_reqs_map.items():
    
    if model_reqs is None:
        AUCs.append(np.nan)
        EERs.append(np.nan)
        HTERs.append(np.nan)
        
        indexes.append(facing_lab)
        
        continue
    
    df_cmp = pd.read_csv('../data/DFIC/partitions/curated_test/' + facing_lab + '/cmp_df.csv')
    df_cmp = df_cmp.loc[df_cmp.subj_id.isin(test_filt_subjs), :]
    df_uncmp = pd.read_csv('../data/DFIC/partitions/curated_test/' + facing_lab + '/uncmp_df.csv')
    df_uncmp = df_uncmp.loc[df_uncmp.subj_id.isin(test_filt_subjs), :]
    
    uncmp_im_paths_req = df_info_test.loc[df_info_test.im_path.isin(df_uncmp.im_path), 'im_path'].to_list()
    uncmp_scos = df_scos_test.loc[df_scos_test.im_path.isin(uncmp_im_paths_req), model_reqs].to_numpy().mean(axis = 1)
    
    cmp_im_paths_req = df_info_test.loc[df_info_test.im_path.isin(df_cmp.im_path), 'im_path'].to_list()
    cmp_scos = df_scos_test.loc[df_scos_test.im_path.isin(cmp_im_paths_req), model_reqs].to_numpy().mean(axis = 1)
    


    
    cmp_scos = cmp_scos[~np.isnan(cmp_scos)]
    uncmp_scos = uncmp_scos[~np.isnan(uncmp_scos)]
    para = 1
    
    
    labs_cmp = np.ones(len(cmp_scos)).astype('int')
    labs_uncmp = np.zeros(len(uncmp_scos)).astype('int')
    
    all_preds = np.concatenate([cmp_scos, uncmp_scos])
    all_labs = np.concatenate([labs_cmp, labs_uncmp])
    
    auc, eer, hter = get_eer(all_labs, all_preds)
    

    AUCs.append(auc)
    EERs.append(eer)
    HTERs.append(hter)
    
    
    indexes.append(facing_lab)
    
    para = 1
    
indexes.append('AVG')
AUCs.append(np.array(AUCs).mean())
EERs.append(np.array(EERs).mean())
HTERs.append(np.array(HTERs).mean())


    
df_final = pd.DataFrame(data = {'AUC': AUCs, 'EER': EERs, 'HTER': HTERs}, index = indexes)
    
table_form = tabulate(df_final, headers = 'keys', tablefmt = 'fancy_grid')
print(table_form)
