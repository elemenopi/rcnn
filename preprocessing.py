#"added comment"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2,keras
import constants
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def needs_resizing(image, target_size):
    """Check if the image needs resizing to the target size."""
    height, width = image.shape[:2]
    target_height, target_width = target_size
    return height < target_height or width < target_width
def resize_with_padding(image, target_size):
    """
    Resize the image to the target size while preserving aspect ratio.
    If the image is smaller than the target size, pad it with zeros.
    """
    # Determine scaling factor for resizing
    height, width = image.shape[:2]
    target_height, target_width = target_size
    scaling_factor = min(target_width / width, target_height / height)
    
    # Resize the image while preserving aspect ratio
    resized = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # Calculate padding
    pad_height = target_height - resized.shape[0]
    pad_width = target_width - resized.shape[1]
    
    # Pad the image with zeros if necessary
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded = cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    
    return padded


def get_iou(bb1, bb2):
  # assuring for proper dimension.
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
  # calculating dimension of common area between these two boxes.
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
  # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
  # calculating intersection area.
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
  # individual areas of both these bounding boxes.
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
  # union area = area of bb1_+ area of bb2 - intersection of bb1 and bb2.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou2(bb1,bb2):
    assert bb1['x1']<bb1['x2']
    assert bb1['y1']<bb1['y2']
    assert bb2['x1']<bb2['x2']
    assert bb2['y1']<bb2['y2']
    #intersection area square coordinates
    x_left = max(bb1['x1'],bb2['x1'])
    x_right = min(bb1['x2'],bb2['x2'])
    y_bottom = min(bb1['y2'],bb2['y2'])
    y_top = max(bb1['y1'],bb2['y1'])

    if x_right < x_left or y_bottom<y_top:
        return 0.0
    intersection_area = (x_right - x_left)*(y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1'])*(bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1'])*(bb2['y2'] - bb2['y1'])
    iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
    assert iou >=0.0
    assert iou <=1.0
    return iou

path = constants.path
annot = constants.annot
index = 160
def showrectinimage(filename):
    img = cv2.imread(os.path.join(path,filename))
    df = pd.read_csv(os.path.join(annot,filename.replace(".jpg",".csv"))) 

    for row in df.iterrows():
        index,data = row
        #print(f'the row is {data.iloc[0]}')
        x1 = int(data.iloc[0].split(" ")[0])
        y1 = int(data.iloc[0].split(" ")[1])
        x2 = int(data.iloc[0].split(" ")[2])
        y2 = int(data.iloc[0].split(" ")[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    plt.figure()
    plt.imshow(img)
    plt.show()

filename = "airplane_"+str(index)+".jpg"
showrectinimage(filename)
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
plt.imshow(imOut)
plt.show()
train_images = []
train_labels = []
num_batches = constants.num_batches
batch_size = len(os.listdir(annot)) // num_batches
for batch_idx in range(num_batches):
    # for every file in the annotation directory
    train_images.clear()
    train_labels.clear()
    for e,i in enumerate(os.listdir(annot)[batch_idx*batch_size:(batch_idx + 1)*batch_size]):
        try:
            if i.startswith("airplane"):
                filename = i.split(".")[0]+".jpg"
                #showrectinimage(filename)
                image = cv2.imread(os.path.join(path,filename))
                df = pd.read_csv(os.path.join(annot,i))
                gtvalues=[]
                for row in df.iterrows():
                    index,data = row
                    x1 = int(data.iloc[0].split(" ")[0])
                    y1 = int(data.iloc[0].split(" ")[1])
                    x2 = int(data.iloc[0].split(" ")[2])
                    y2 = int(data.iloc[0].split(" ")[3])
                    gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
                ss.setBaseImage(image)   # setting given image as base image
                ss.switchToSelectiveSearchFast()     # running selective search on bae image 
                ssresults = ss.process()     # processing to get the outputs
                imout = image.copy()  
                counter = 0
                falsecounter = 0
                flag = 0
                fflag = 0
                bflag = 0
                data = []
                for k,result in enumerate(ssresults):
                    if k < 2000 and flag == 0:     # till 2000 to get top 2000 regions only
                        for gtval in gtvalues:
                            x,y,w,h = result
                            if (w<30 and h<30):
                                continue
                            #data.append({"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                            iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})  # calculating IoU for each of the proposed regions
                            if counter < 25:       # getting only 30 psoitive examples
                                if iou > 0.75:     # IoU or being positive is 0.8
                                    x1 = max(0, x - 10)
                                    y1 = max(0, y - 10)
                                    x2 = min(imout.shape[1], x + w + 10)
                                    y2 = min(imout.shape[0], y + h + 10)
                                    # Extract the region with adjusted coordinates
                                    timage = imout[y1:y2, x1:x2]
                                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(1)
                                    counter += 1
                            else :
                                fflag =1              # to insure we have collected all psotive examples
                            if falsecounter <25:      # 30 negatve examples are allowed only
                                if iou < 0.25:         # IoU or being negative is 0.3
                                    x1 = max(0, x - 10)
                                    y1 = max(0, y - 10)
                                    x2 = min(imout.shape[1], x + w + 10)
                                    y2 = min(imout.shape[0], y + h + 10)
                                    # Extract the region with adjusted coordinates
                                    timage = imout[y1:y2, x1:x2]
                                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(0)
                                    falsecounter += 1
                            else :
                                bflag = 1             #to ensure we have collected all negative examples
                        if fflag == 1 and bflag == 1:  
                            flag = 1        # to signal the complition of data extaction from a particular image
                #df2 = pd.DataFrame(data,columns = ['x1','x2','y1','y2'])
                #csv_filename = f"file{e}.csv"
                #df2.to_csv(csv_filename,index = False)
        except Exception as e:
            print(e)
            print("error in "+filename)
            continue 
    
    X_new = np.array(train_images)
    Y_new = np.array(train_labels)
    folder_name = "preprocessing_results"
    os.makedirs(folder_name,exist_ok = True)
    np.save(os.path.join(folder_name,f"X_new_{batch_idx}.npy"),X_new)
    np.save(os.path.join(folder_name,f"Y_new_{batch_idx}.npy"),Y_new)

#idxone = []
#idxzero = []
#for idx,data in enumerate(train_labels):
#    if data == 1:
#        idxone.append(idx)
#    else:
#        idxzero.append(idx)
#plt.imshow(X_new[0])
#plt.plot()
#print(tf.__version__)