# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:43:32 2021

@author: Yokhesh
"""
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings("ignore")

trained_face_det = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/trained_res_ssd.caffemodel")

facial_landmark = tf.saved_model.load('models/pose_model')
cap = cv2.VideoCapture('video_test.mp4');
#cap = cv2.VideoCapture(0);
ret, img = cap.read()
tot_wid = img.shape[1]
center = (img.shape[1]/2, img.shape[0]/2)
cam_par = np.array([[tot_wid, 0, center[0]],[0, tot_wid, center[1]],[0, 0, 1]], dtype = "double")
theta_lst = [];
phi_lst = [];
while True:
    ret, img = cap.read()
    if ret == True:
        ###############################################
        #img = cv2.resize(img,(1000,1000))        
        mean_f_blob =  (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (480, 240)), 1.0,(480, 240),mean_f_blob)
        trained_face_det.setInput(blob)
        for_prop = trained_face_det.forward()
        faces = []
        center_list = [];
        width_height_lst = [];
        h, w = img.shape[:2]
        image_center = [w,h]
        for i in range(for_prop.shape[2]):
            if for_prop[0,0,i,2] > 0.5:
                (x, y, x1, y1) = (for_prop[0,0,i,3:7] * np.array([w, h, w, h])).astype("int")
                faces.append([x, y, x1, y1])
        ###############################################
        if len(faces) != 0:
            for face in faces:
                x,y, width, height = face
                x2, y2 = x + width, y + height
                x_center = x + width/2;y_center = y + height/2;
                center = [x_center,y_center]
                center_list.append(center)
                #wid_hei = width+height;
                wid_hei = (face[2]-face[0])+(face[3]- face[1]);
                width_height_lst.append(wid_hei)
        
            center_arr = np.array(center_list)
            image_cen_arr = np.array(image_center)
            width_height_arr = np.array(width_height_lst)
            h_dis = (center_arr-image_cen_arr)**2
            h_dis_sum = h_dis[:,0] + h_dis[:,1]
            closest_ind = np.where(h_dis_sum == h_dis_sum.min())
            closest_ind = int(closest_ind[0])
            
            closest_ind1 = np.where(width_height_arr == width_height_arr.max())
            closest_ind1 = int(closest_ind1[0])
            face_count = 0;
        
        for face in faces:
                    ###############################################   
            x = face[0];y = face[1] + int(abs((face[3] - face[1]) * 0.1))
            x1 = face[2];y1= face[3] + int(abs((face[3] - face[1]) * 0.1))

            wid_hei_equ = (y1 - y) - (x1 - x)
            compensat = int(abs(wid_hei_equ) / 2)
            if wid_hei_equ == 0:              
                yu = 1;
            elif wid_hei_equ > 0:              
                x -= compensat
                x1 += compensat
                if wid_hei_equ % 2 == 1:
                    x1 += 1
            else:                    
                y -= compensat
                y1 += compensat
                if wid_hei_equ % 2 == 1:
                    y1 += 1
            face = [x, y, x1, y1]
                    ###############################################   
            face_update = face;
            
            if face_count == closest_ind1:
                cv2.rectangle(img,(face_update[0], face_update[1]),(face_update[2], face_update[3]), (0,255,0), 3)
                roi = img[face_update[1]: face_update[3],face_update[0]: face_update[2]]
                roi = cv2.resize(roi, (128, 128))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi= roi.astype(np.uint8)
                
                landmark_pred = facial_landmark.signatures["predict"]

                arr4d = np.expand_dims(roi, 0)
            
                fg = tf.constant(arr4d,dtype = tf.uint8)
                
                landmark_pred = landmark_pred(fg)
                landmarks = np.array(landmark_pred['output']).flatten()[:136]
                landmarks = np.reshape(landmarks, (-1, 2))
                
                landmarks *= (face_update[2] - face_update[0])
                landmarks[:, 0] = landmarks[:, 0] + face_update[0]
                landmarks[:, 1] = landmarks[:, 1] + face_update[1]
                landmarks = landmarks.astype(np.uint)
            ###############################################
                identified_landmarks = np.array([landmarks[30],landmarks[8],landmarks[36],landmarks[45],landmarks[48],landmarks[54]], dtype="double")
                ###########################################
                lens_distortion = np.zeros((4,1))
                ##############################################
                model_points = np.array([(0, 0.0, 0.0),(0.0, -330.0,0),(-225.0, 170.0, 0),(225.0, 170.0, 0),(-150.0, -150.0, 0),(150.0, -150.0, 0)])
                (_, rv, tv) = cv2.solvePnP(model_points, identified_landmarks, cam_par, lens_distortion, flags=cv2.SOLVEPNP_UPNP)
                ###############################################
                (nose_center, jacobian_matrix) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rv, tv, cam_par, lens_distortion)
                ###############################################
                cv2.circle(img, (int(identified_landmarks[0][0]), int(identified_landmarks[0][1])), 10, (0,0,255), 2)
                cv2.circle(img, (int(identified_landmarks[1][0]), int(identified_landmarks[1][1])), 10, (0,0,255), 2)
                cv2.circle(img, (int(identified_landmarks[2][0]), int(identified_landmarks[2][1])), 10, (0,0,255), 2)
                cv2.circle(img, (int(identified_landmarks[3][0]), int(identified_landmarks[3][1])), 10, (0,0,255), 2)
                cv2.line(img,(int(identified_landmarks[4][0]), int(identified_landmarks[4][1])), (int(identified_landmarks[5][0]), int(identified_landmarks[5][1])),(0,0,255), 5)
                ############################################
                val = [1, 0, img.shape[1], img.shape[1]*2]
                camera3 = []
                camera3.append((-1, -1, 0));camera3.append((-1, 1, 0));camera3.append((1, 1, 0));camera3.append((1, -1, 0));camera3.append((-1, -1, 0))
                camera3.append((-img.shape[1], -img.shape[1], img.shape[1]*2));camera3.append((-img.shape[1], img.shape[1], img.shape[1]*2));camera3.append((img.shape[1], img.shape[1], img.shape[1]*2));camera3.append((img.shape[1], -img.shape[1], img.shape[1]*2))
                camera3.append((-img.shape[1], -img.shape[1], img.shape[1]*2));camera3 = np.array(camera3, dtype=np.float).reshape(-1, 3)
                (camera2,_) = cv2.projectPoints(camera3,rv,tv,cam_par,lens_distortion);camera2 = np.int32(camera2.reshape(-1, 2))
                up_down_det1 = ( int(identified_landmarks[0][0]), int(identified_landmarks[0][1]))
                up_down_det2 = (int(nose_center[0][0][0]),int(nose_center[0][0][1]))
                right_left_det2 = (camera2[5] + camera2[8])//2
                right_left_det1 = camera2[2]
				
                try:
                    m = (up_down_det2[1] - up_down_det1[1])/(up_down_det2[0] - up_down_det1[0])
                    angle1 = int(math.degrees(math.atan(m)))
                except:
                    angle1 = 90
                    
                try:
                    m = (right_left_det2[1] - right_left_det1[1])/(right_left_det2[0] - right_left_det1[0])
                    angle2 = int(math.degrees(math.atan(-1/m)))
                except:
                    angle2 = 90
                ############################################
                font = cv2.FONT_HERSHEY_SIMPLEX 
                angle1_str = "theta = "+str(angle1)
                angle2_str = "Phi = "+str(angle2)
                cv2.putText(img, str(angle1), tuple(up_down_det1), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 255, 0), 3)
                cv2.putText(img, str(angle2), tuple(right_left_det1),cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,255), 3)
                print(angle1_str)
                print(angle2_str)
                theta_lst.append(angle1)
                phi_lst.append(angle2)
                ############################################
                face_count = face_count+1;
            else:
                cv2.rectangle(img,(face_update[0], face_update[1]),(face_update[2], face_update[3]), (0,0,255), 3)
                face_count = face_count+1;
        cv2.imshow('img', img)
        if cv2.waitKey(25) == 13:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
            
            
            