
import cv2 


class YuNetFaceDetector():
    
    def __init__(self):
        

        yunet = cv2.FaceDetectorYN.create(
        model='method/face_detection/face_detection_yunet_2023mar.onnx',
        config='',
        input_size=(320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        # top_k=1,
        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        self.YN = yunet
    
    def extract_face_feats(self,im_CV_BGR):
        
    
        image = im_CV_BGR
        width=image.shape[1]
        height=image.shape[0]
        
        if width<=height:
            w=320
            h= int(height*(w/width))
        else:
            h = 320
            w= int(width*(h/height))
            
        image = cv2.resize(image, (w,h))
        self.YN.setInputSize((image.shape[1], image.shape[0]))
        dec, faces = self.YN.detect(image)
        
        
        
        bbox_results = []
        if faces is not None:    
            
            for face_iter in range(len(faces)):
                face = faces[face_iter].tolist()
                
                # bounding box
                f0 = int(face[0]*(width / w))
                f1 = int(face[1]*(height / h))
                f2 = int(face[2]*(width / w))
                f3 = int(face[3]*(height / h))
                
                #left eye
                f4 = int(face[4]*(width / w))
                f5 = int(face[5]*(height / h))
                
                # right eye
                f6 = int(face[6]*(width / w))
                f7 = int(face[7]*(height / h))
                
                # nose
                f8 = int(face[8]*(width / w))
                f9 = int(face[9]*(height / h))
                
                # left mouth
                f10 = int(face[10]*(width / w))
                f11 = int(face[11]*(height / h))
                
                # right mouth
                f12 = int(face[12]*(width / w))
                f13 = int(face[13]*(height / h))
                
                bbox_results.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, face[14]])
            
            return bbox_results
        else:
            return None
        

    

        

