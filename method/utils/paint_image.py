#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:27:31 2025

@author: visteamer
"""

import cv2
import numpy as np


def fill_module(im_cv_bgr, text_dict, x_init, y_init, sup_labs):
    
    y_step = 13
    y_curr = y_init
    index_counter = 0
    for k, v in text_dict.items():
        
        if k == 'title':
            msg_line = v
        else:
            msg_line = k + ' ' + str(v)
            
        if index_counter > 0 and sup_labs[index_counter-1] == 0:
            color = (0, 128, 128)
        else:
            
            if isinstance(v, str):
                color = (128,0,0)   
            elif v <= 0.5:
                color = (0,0,128)
            else:
                color = (0,128,0)
            
        im_cv_bgr = cv2.putText(im_cv_bgr, msg_line, (x_init, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
        y_curr = y_curr + y_step
        
        index_counter += 1
        
    return im_cv_bgr
    
    

def fill_info(im_cv_bgr, scores, sup_labels):
    
    sup_labels = np.asarray(sup_labels)

    y_step = 13
    # yv, xv = np.meshgrid(list(range(masks.shape[1])), list(range(masks.shape[2])), indexing='ij')
    
    ### Eye/ SunGlasses Module 
    loc4_eye_info_x = 340 # xv[masks[3]].max() + 5
    loc4_eye_info_y = 170 # yv[masks[3]].min() + 15
    
    text_eyes = {'title': 'Eyes/Sun Reqs:',
                 '(1) Open:': round(scores[0], 3),
                 '(6) Forward Look:': round(scores[5], 2),
                 '(7) No Hair Occ:': round(scores[6], 2),
                 '(11) No Shades:': round(scores[10], 2),
                 '(12) No Frame Occ:': round(scores[11], 2),
                 '(13) No Lense Refl:': round(scores[12], 2),
                 '(14) No Heavy Frame:': round(scores[13], 2),
                 '(19) Red Eyes:': round(scores[18], 2),}
    
    sup_labels_eyes = sup_labels[[0, 5, 6, 10, 11, 12, 13, 18]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_eyes, loc4_eye_info_x, loc4_eye_info_y - (y_step*len(text_eyes))//2, sup_labels_eyes)
    
    ### Mouth Module 
    loc4_mouth_info_x = 320 # xv[masks[4]].max() + 5
    loc4_mouth_info_y = 290 # int(yv[masks[4]].mean())
    
    text_mouth = {'title': 'Mouth Reqs:',
                 '(3) Closed:': round(scores[2], 2),
                 '(9) No Mouth Occ:': round(scores[8], 2)}
    
    sup_labels_mouth = sup_labels[[2, 8]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_mouth, loc4_mouth_info_x, loc4_mouth_info_y - (y_step*len(text_mouth))//2, sup_labels_mouth)

    
    ### Face Module 
    loc4_face_info_x = 10
    loc4_face_info_y = 300 # int(yv[masks[5]].mean())
    
    text_face = {'title': 'Face Reqs:',
                 '(2) Neutral Exp:': round(scores[1], 2),
                 '(5) Frontral Face:': round(scores[4], 2),
                 '(16) No Face Shadows:': round(scores[15], 2),
                 '(17) Flash Face Refl:': round(scores[16], 2),
                 '(18) Natural Skin Tone:': round(scores[17], 2),
                 '(20) Too Dark/Light:': round(scores[19], 2),
                 '(21) No Blur:': round(scores[20], 2),}
    
    sup_labels_face = sup_labels[[1, 4, 15, 16, 17, 19, 20]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_face, loc4_face_info_x, loc4_face_info_y - (y_step*len(text_face))//2, sup_labels_face)
        

    ### Body Module 
    loc4_body_info_x = 256
    loc4_body_info_y = 512 - 50

    text_body = {'title': 'Body Reqs:',
                 '(4) Aligned Shoulders:': round(scores[3], 2),
                 '(10) No Objects:': round(scores[9], 2)}
    
    sup_labels_body = sup_labels[[3, 9]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_body, loc4_body_info_x, loc4_body_info_y - (y_step*len(text_body))//2, sup_labels_body)
        

    ### Back Module 
    loc4_back_info_x = 10
    loc4_back_info_y = 30

    text_back = {'title': 'Background Reqs:',
                 '(15) Shadows Behind:': round(scores[14], 2),
                 '(22) Varied Background:': round(scores[21], 2),
                 '(23) No Pixelization:': round(scores[22], 2),
                 '(24) No Washed:': round(scores[23], 2),
                 '(25) No Ink/Mark/Crease:': round(scores[24], 2),
                 '(26) No Posterization:': round(scores[25], 2),}
    
    sup_labels_back = sup_labels[[14, 21, 22, 23, 24, 25]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_back, loc4_back_info_x, loc4_back_info_y, sup_labels_back) #  - (y_step*len(text_back))//2
        
    
    ### Head Coverings Module 
    loc4_hc_info_x = 256 + 30
    loc4_hc_info_y = 30

    text_hc = {'title': 'Head Covering Req:',
                 '(8) No Uncmp Head Covering:': round(scores[7], 2)}
    
    sup_labels_hc = [sup_labels[7]]
    
    im_cv_bgr = fill_module(im_cv_bgr, text_hc, loc4_hc_info_x, loc4_hc_info_y - (y_step*len(text_hc))//2, sup_labels_hc)
        



    return im_cv_bgr


def get_ini_msg_pane(im2show, atributes, image_attributes, letter_reqindex_map):
    
    reqindex_letter_map = {v: k for k, v in letter_reqindex_map.items()}
    
    inter_pane = np.zeros([700,600,3],dtype=np.uint8)
    inter_pane.fill(255)
    
    ini_msg1 = "Correct Automatic Classifications"
    ini_msg2 = "    Press the corresponding key letter to correct"
    ini_msg3 = "    Press ENTER if everything seems good and defined"
    ini_msg4 = "    Press ESC to stop the annotation process"
    ini_msg5 = "    Press SPACE to skip this annotation"
    
    inter_pane = cv2.putText(inter_pane, ini_msg1, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    inter_pane = cv2.putText(inter_pane, ini_msg2, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    inter_pane = cv2.putText(inter_pane, ini_msg3, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    inter_pane = cv2.putText(inter_pane, ini_msg4, (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    inter_pane = cv2.putText(inter_pane, ini_msg5, (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    
    
    status_pane = np.zeros([700,300,3],dtype=np.uint8)
    status_pane.fill(255)
    status_pane = cv2.putText(status_pane, 'Status', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    counter = 0
    reqs2define = []
    reqs2define_inds = []
    for req_name, req_key_ann in image_attributes.items():
        
        if req_key_ann == 'zzz':
            reqs2define.append(req_name)
            reqs2define_inds.append(counter)
            key_descrip = 'To Be Defined'
            
        else:
            for opt_ind, opt in enumerate(atributes[req_name]["options"]):
                if opt['key'] == req_key_ann:
                    key_descrip = atributes[req_name]["options"][opt_ind]["description"]
                
        if req_key_ann != '0':
            req_col = (0,0,128)
        else:
            req_col = (0,0,0)

        msg_stat_i = '(' + reqindex_letter_map[counter] + ') ' + req_name + " - " + key_descrip
        status_pane = cv2.putText(status_pane, msg_stat_i, (20, 200 + 15*(counter)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, req_col, 1, cv2.LINE_AA)
        counter += 1
        
        
    msg5 = "You must define the following requirements:"
    inter_pane = cv2.putText(inter_pane, msg5, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if len(reqs2define) != 0:
        for req_i, req in enumerate(reqs2define):
            msg_req = ' - (' + reqindex_letter_map[reqs2define_inds[req_i]] + ') ' + req
            inter_pane = cv2.putText(inter_pane, msg_req, (20, 370 + 20*(req_i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    join_im = np.concatenate((inter_pane, im2show, status_pane), axis = 1)
    
    return join_im


def get_req_msg(im2show, atributes, req_name, image_attributes, letter_reqindex_map, cam_im, cam_labels):
    
    reqindex_letter_map = {v: k for k, v in letter_reqindex_map.items()}
    
    inter_pane = np.zeros([700,600,3],dtype=np.uint8)
    inter_pane.fill(255)
    
    inter_pane = cv2.putText(inter_pane, atributes[req_name]["showtext"], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    cam_im_res = cv2.resize(cam_im,(int(cam_im.shape[1]*(float(298)/float(cam_im.shape[0]))),298))
    
    inter_pane[400:698, 300:598] = cam_im_res
    
        
    cam_req_key = cam_labels[req_name]

    optkey_char_map = {}
    for optind, optkey in enumerate(atributes[req_name]["options"]):
        optkey_char_map[ord(optkey["key"])] = optkey["key"]
        
        if optkey["key"] == cam_req_key:
            color = (0, 128, 128)
        else:
            color = (0,0,0)
        inter_pane = cv2.putText(inter_pane, str(optkey["key"] + " - " + optkey["description"]), (20, 150+20*(1+optind)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    
    

    status_pane = np.zeros([700,300,3],dtype=np.uint8)
    status_pane.fill(255)
    status_pane = cv2.putText(status_pane, 'Status', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    counter = 0
    for req_name, req_key_ann in image_attributes.items():
        
        if req_key_ann == 'zzz':
            key_descrip = 'To Be Defined'
            
        else:
            for opt_ind, opt in enumerate(atributes[req_name]["options"]):
                if opt['key'] == req_key_ann:
                    key_descrip = atributes[req_name]["options"][opt_ind]["description"]
                    break
            
        if req_key_ann != '0':
            req_col = (0,0,128)
        else:
            req_col = (0,0,0)
            
            
        msg_stat_i = '(' + reqindex_letter_map[counter] + ') ' + req_name + " - " + key_descrip

        status_pane = cv2.putText(status_pane, msg_stat_i, (20, 200 + 15*(counter)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, req_col, 1, cv2.LINE_AA)
        counter += 1
    
    join_im = np.concatenate((inter_pane, im2show, status_pane), axis = 1)
    
    
    return join_im, optkey_char_map
