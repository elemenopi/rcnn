import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2,keras

def get_iou(bb1,bb2):
    assert bb1['x1']<bb1['x2']
    assert bb1['y1']<bb1['y2']
    assert bb2['x1']<bb2['x2']
    assert bb2['y1']<bb2['y2']
    #intersection area square coordinates
    x_left = max(bb1['x1'],bb2['x1'])
    x_right = min(bb1['x2'],bb2['x2'])
    y_bottom = min(bb1['y2'],bb2['y2'])
    y_top = max(bb1['y1'],bb2['x1'])

    if x_right < x_left or y_bottom<y_top:
        return 0.0
    intersection_area = (x_right - x_left)*(y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1'])*(bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1'])*(bb2['y2'] - bb2['y1'])
    iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
    assert iou >=0.0
    assert iou <=1.0
    return iou

path = "Images"
annot = "Annotations"
index = 148
filename = "airplane_"+str(index)+".jpg"
img = cv2.imread(os.path.join(path,filename))
df = pd.read_csv(os.path.join(annot,filename.replace(".jpg",".csv"))) 
for row in df.iterrows():
    index,data = row
    print(f'the row is {data.iloc[0]}')
    x1 = int(data.iloc[0].split(" ")[0])
    y1 = int(data.iloc[0].split(" ")[1])
    x2 = int(data.iloc[0].split(" ")[2])
    y2 = int(data.iloc[0].split(" ")[3])
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.figure()
#plt.imshow(img)
#plt.show()
print("we are here")
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
im = cv2.imread(os.path.join(path,"42850.jpg"))
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()
rects = ss.process()
imOut = im.copy()
for i, rect in enumerate(rects):
    x,y,w,h = rect
    cv2.rectangle(imOut,(x,y),(x+w,y+h),(0,255,0),1,cv2.LINE_AA)
train_images = []
train_labels = []
# for every file in the annotation directory
for e,i in enumerate(os.listdir(annot)):
    if e == 100:
        break
    try:
        #if it starts with airplane
        if i.startswith("airplane"):
            filename = i.split(".")[0]+".jpg"
            image = cv2.imread(os.path.join(path,filename))
            df = pd.read_csv(os.path.join(annot,i))
            gtvalues = []
            #put the ground truth squares surrounding inside gtvalues
            for row in df.iterrows():
                index,data = row
                x1 = int(data.iloc[0].split(" ")[0])
                y1 = int(data.iloc[0].split(" ")[1])
                x2 = int(data.iloc[0].split(" ")[2])
                y2 = int(data.iloc[0].split(" ")[3])
                gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            #ssresult contains an array st each row is [x,y,w,h] of estimated ss algorithm squares
            ssresult = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e,result in enumerate(ssresult):
                if e<2000 and flag ==0:
                    for gtval in gtvalues:
                        x,y,w,h = result
                        print(gtval)
                        iou = get_iou(gtval,{'x1':x,'x2':x+w,'y1':y,'y2':y+h})     
                        if counter <30:
                          if iou >0.70:
                                timage = imout[x:x+w,y,y+h]
                                resized = cv2.resize(timage,(224,224),interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter+=1
                        else:
                            fflag = 1
                        if falsecounter<30:
                            if iou < 0.3:
                                timage = imout[x:x+w,y:y+h]
                                resized = cv2.resize(timage,(224,224),interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else:
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        continue    

X_new = np.array(train_images)
Y_new = np.array(train_labels)
print(Y_new)
#plt.imshow(X_new[0])
#plt.plot()
#print(tf.__version__)