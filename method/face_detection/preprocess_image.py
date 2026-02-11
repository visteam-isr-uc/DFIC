
from method.face_detection.yu_net_wrapper import YuNetFaceDetector

import numpy as np
from PIL import Image


class Preprocess_Image:
    
    def __init__(self):
        

        self.yn_det = YuNetFaceDetector()



    def get_image_scaled(self, im_cv_bgr):
        
        bboxes = self.yn_det.extract_face_feats(im_cv_bgr)

        
        if bboxes is not None:

            
            bboxes_xywh = [bb[:4] for bb in bboxes]
            
            im_c = [im_cv_bgr.shape[1]//2, im_cv_bgr.shape[0]//2]
            
            bboxes_dist2center = np.array([np.sqrt((bb[0]+bb[2]//2 - im_c[0])**2 + (bb[1]+bb[3]//2 - im_c[1])**2) for bb in bboxes_xywh])
            ord_indexes = bboxes_dist2center.argsort()
            
            best_bb = bboxes_xywh[ord_indexes[0]]
            
            y1 = best_bb[1]-int(best_bb[3]*0.7)
            y2 = best_bb[1]+int(best_bb[3]*2)
            h = y2-y1
            x1 = (best_bb[0]+best_bb[2]//2) - h//2
            x2 = (best_bb[0]+best_bb[2]//2) + h//2
            
            if (im_cv_bgr.shape[1] * im_cv_bgr.shape[0]) < h**2:
                output = im_cv_bgr.copy()
                crop_bb_xyxy = [0,0,im_cv_bgr.shape[1]-1, im_cv_bgr.shape[0]-1]
            else:
                im_pil = Image.fromarray(im_cv_bgr)
                im_pil_cropped = im_pil.crop((x1, y1, x2, y2))
                output = np.array(im_pil_cropped)
                crop_bb_xyxy = [x1, y1, x2, y2]

            
            return crop_bb_xyxy, output
        
        return None, None