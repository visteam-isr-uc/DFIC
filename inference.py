

import argparse
import pandas as pd
import cv2
import numpy as np
import sys
import os
from tabulate import tabulate
from tqdm import tqdm

from method.face_detection.preprocess_image import Preprocess_Image
from method.icao_model_wrapper_4inference import ICAO_Model


def parse_args():
    
    parser = argparse.ArgumentParser(description = 'Script to perform ICAO inference on image files') 
    
    parser.add_argument('--model_path', default='method/models/DFIC_best.pth.tar', type=str, 
                        help='Model file to use. The options are in the method/models/ directory')
    parser.add_argument('--image_path', default=None, type=str, help='path to the image to perform inference. The path can be absolute, or relative to this directory')
    parser.add_argument('--image_list_path', default=None, type=str, help='path to .txt containing the image paths to perform inference. The paths can be absolute, or relative to this directory')
    parser.add_argument('--output_csv_path', default='inference_out.csv', type=str, help='csv output file destination. Only used in case of image_path_list is defined.')
    parser.add_argument('--batch_inference', default=False, type=bool, help='Whether to perform bacth inference for large quantity of images (faster), or not. Default is False')
    parser.add_argument('--device', type=str, default='cpu', help='Device id. Define the GPU ID if GPU inference is needed. CPU is the default')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='Batch size for batch inference')
    parser.add_argument('--num_workers', '-nw', default=8, type=int, help='Number of workers for batch inference')


    args = parser.parse_args()
      
    return args



if __name__ == '__main__':

    
    args = parse_args()
    
    
    
    if (args.image_path is None) and (args.image_list_path is None):
        print('One of the arguments: --image_path or --image_list_path, must be defined.')
        sys.exit()
        
    if (args.image_path is not None) and (args.image_list_path is not None):
        print('ONLY one of the arguments:  --image_path or --image_list_path, must be defined. Not Both.')
        sys.exit()
        
        
    face_scaled_crop_processor = Preprocess_Image()
    icao_model = ICAO_Model(model_path = args.model_path)
        
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
    
    
    if args.image_path is not None:
        
        if not os.path.exists(args.image_path):
            print('The specified image file: {} does not exists.'.format(args.image_path))
            sys.exit()
        
        im_cv_bgr = cv2.imread(args.image_path)
        
        
        
        _, prep_crop_image = face_scaled_crop_processor.get_image_scaled(im_cv_bgr)
        if prep_crop_image is None:
            print('Unable to detect a face in the image file: {}.'.format(args.image_path))
            
        else:
            _, final_preds = icao_model.process(prep_crop_image)
            
            dict_scos = {names[i]: [round(1-final_preds[i], 3)] for i in range(len(names))}
            df_res = pd.DataFrame(dict_scos).T
            df_res.columns = ['Uncompliance Prob.']
            
            print('ICAO requirements compliance probabilities for image {}:'.format(args.image_path))
            table_form = tabulate(df_res, headers = 'keys', tablefmt = 'fancy_grid')
            print(table_form)
            
            
    if args.image_list_path is not None:
        
        if not os.path.exists(args.image_list_path):
            print('The specified list file: {} does not exists.'.format(args.image_list_path))
            sys.exit()
            
        image_list_paths = pd.read_csv(args.image_list_path, header = None)[0]
        
        
        predictions = []
        im_paths = []
        face_detected = []
        im_found = []
        for im_p in tqdm(image_list_paths):
            
            if not os.path.exists(im_p):
                print('The specified image file: {} does not exists.'.format(im_p))
                im_pres = False
                final_preds = np.empty(26)
                final_preds[:] = np.nan
                face_det = False

        
            im_cv_bgr = cv2.imread(im_p)
      
            _, prep_crop_image = face_scaled_crop_processor.get_image_scaled(im_cv_bgr)
            if prep_crop_image is None:
                print('Unable to detect a face in the image file: {}.'.format(im_p))
                final_preds = np.empty(26)
                final_preds[:] = np.nan
                face_det = False
                im_pres = True
                
            else:
                _, final_preds = icao_model.process(prep_crop_image)
                face_det = True
                im_pres = True
                
            predictions.append(final_preds)
            im_paths.append(im_p)
            face_detected.append(face_det)
            im_found.append(im_pres)
                
        predictions_np = np.stack(predictions)
        df_res = pd.DataFrame({'im_path': im_paths,
                               'im_found': im_found,
                               'face_detected': face_detected})
                
        dict_scos = {names[i]: 1-predictions_np[:,i] for i in range(len(names))}
        df_res = pd.concat([df_res, pd.DataFrame(dict_scos)], axis = 1)
        
        df_res.to_csv(args.output_csv_path, index = False)
        

            
        

    
    
