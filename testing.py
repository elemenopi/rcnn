import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2,keras
import constants
path = constants.path
annot = constants.annot
loaded_model = tf.keras.models.load_model("my_model.h5")
image = cv2.imread(os.path.join(path,"airplane_050.jpg"))
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()
imOut = image.copy()
boxes = []
count = 0
for e,result in enumerate(ssresults):
    if e<600:
        x,y,w,h = result
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(imOut.shape[1], x + w + 10)
        y2 = min(imOut.shape[0], y + h + 10)
        # Extract the region with adjusted coordinates
        timage = imOut[y1:y2, x1:x2]
        resized = cv2.resize(timage,(224,224),interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized,axis = 0)
        out = loaded_model.predict(resized)
        if out[0][0]>0.75:
            boxes.append([x,y,w,h])
for box in boxes:
    x,y,w,h = box
    cv2.rectangle(imOut,(x,y),(x+w,y+h),(0, 255, 0), 1, cv2.LINE_AA)
    
plt.imshow(imOut)
plt.show()
        