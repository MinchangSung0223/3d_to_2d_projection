import cv2
import numpy as np
import math
import os


fps_3d = np.load('feature_3d.npy')
'''
fps_3d = np.array([[ 0.020191,  0.033474,  0.020549],
       [-0.020191,  0.033474,  0.020549],
       [ 0.027228,  0.001891,  0.020549],
       [-0.027228,  0.001891,  0.020549],
       [ 0.      , -0.011374,  0.020549],
       [ 0.033196, -0.091085,  0.020549],
       [-0.033196, -0.091085,  0.020549]],dtype=float)
'''
img_list = os.listdir('rgb')

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

cammatrix =np.array([[ 7.000e+02 ,0.000000000000000000e+00, 3.20e+02],
[0.000000000000000000e+00 ,7.00e+02, 2.40e+02],
[0.000000000000000000e+00 ,0.000000000000000000e+00, 1.000000000000000000e+00]],dtype=float)

home_path = os.getcwd()
for name in img_list:
	os.chdir(home_path)
	os.chdir('pose')
	cam2obj = np.load('pose'+str(name[:-4])+'.npy')
	os.chdir(home_path)
	os.chdir('rgb')
	img = cv2.imread(str(name[:-4])+'.jpg')
	os.chdir(home_path)
	os.chdir('mask')
	mask = cv2.imread(str(name[:-4])+'.png')
	mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
	os.chdir(home_path)
	os.chdir('sample')
	fps_2d = project(fps_3d,cammatrix,cam2obj)
	sample = []
	count  = 0
	mask_temp = mask.copy()
	for i in fps_2d:
		print(i)
		h = 0
		w = 0
		sample_img = img.copy()
		mask = mask_temp.copy()
		sample_img = cv2.line(img,(int(i[0]),int(i[1])),(int(i[0])+1,int(i[1])+1),(255,255,0),3)
		sample.append(sample_img)
		count = count +1
	#cv2.imshow('rgb',img)
	print(len(fps_2d))
	for k in range(1):
		cv2.imshow('mask'+str(k), np.uint8(sample[k]))
		cv2.imwrite('sample_'+str(k)+"_"+str(name[:-4])+".png",np.uint8(sample[k]))
		cv2.waitKey(1)

