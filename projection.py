import cv2
import numpy as np
import math
import os
fps_3d = np.load('feature_3d.npy')
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
	os.chdir('vertex')
	fps_2d = project(fps_3d,cammatrix,cam2obj)
	vertex = []
	count  = 0
	mask_temp = mask.copy()
	for i in fps_2d:
		print(i)
		h = 0
		w = 0
		if count >0 : 
			break;
		vertex_img = np.zeros(mask.shape,dtype=np.float)
		mask = mask_temp.copy()
		print(mask[mask>0].shape)
		for h in range(480):
			for w in range(640):
				if mask[h,w]>0:
					vertex_img[h,w] = ((math.atan2(i[1]-h,i[0]-w)+math.pi)/2/math.pi)
					#print(vertex_img[h,w]*255)
					#vertex_img[h,w] = math.sqrt((i[0]-h)*(i[0]-h) + 1.5*(i[1]-w)*(i[1]-w))/5
		vertex.append(vertex_img)
		#print(vertex_img)
		#show_img = cv2.line(img,(int(i[0]),int(i[1])),(int(i[0])+1,int(i[1])+1),(255,0,0),1)
		count = count +1
	#cv2.imshow('rgb',img)
	print(len(fps_2d))
	for k in range(1):
		#cv2.imshow('mask'+str(k), np.uint8(vertex[k]*255))
		cv2.imwrite('vertex_'+str(k)+"_"+str(name[:-4])+".png",np.uint8(vertex[k]*255))
	#cv2.waitKey(0)


