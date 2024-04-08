import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2,keras
import constants
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
path = constants.path
annot = constants.annot
X_new = np.load("preprocessing_results/X_new_2.npy")
Y_new = np.load("preprocessing_results/Y_new_2.npy")
num_images = 25
count = 0
one_labels = []
for i in range(len(Y_new)):
    if Y_new[i] == 0:
        one_labels.append(i)
    
# Display images and corresponding labels
fig, axes = plt.subplots(5, 5, figsize=(15, 9))  # Adjust the size of the figure as needed

for i in range(num_images):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_new[one_labels[i]])
    ax.set_title(f"Label: {one_labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
base_model = tf.keras.applications.MobileNetV2(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
base_model.trainable = False
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
predictions = tf.keras.layers.Dense(1,activation = 'sigmoid')(x)

model = tf.keras.models.Model(inputs = base_model.input,outputs = predictions)
print(base_model.input)
model.summary()
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(X_new,Y_new,batch_size = 64,epochs = 3,verbose = 1,validation_split= 0.05,shuffle = True)
model.save("my_model.h5")
##################################

#vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#for layer in vgg.layers[:-2]:
#  layer.trainable = False
#x = vgg.get_layer('fc2')
#last_output =  x.output
#x = tf.keras.layers.Dense(1,activation = 'sigmoid')(last_output)  
#model = tf.keras.Model(vgg.input,x)
#model.compile(optimizer = "adam", 
#              loss = 'binary_crossentropy', 
#              metrics = ['acc'])
#model.summary()
#model.fit(X_new,Y_new,batch_size = 64,epochs = 3, verbose = 1,validation_split=.05,shuffle = True)


##model test
#image = cv2.imread(os.path.join(path,'airplane_020.jpg'))
#ss.setBaseImage(image)
#ss.switchToSelectiveSearchFast()
#ssresults = ss.process()
#
#imOut = image.copy()
#boxes = []
#count = 0
#for e,result in enumerate(ssresults):
#  if e < 50:
#    x,y,w,h = result
#    timage = imOut[x:x+w,y:y+h]
#    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
#    resized = np.expand_dims(resized,axis = 0)
#    out = model.predict(resized)
#    print(e,out)
#    if(out[0][0]<out[0][1]):
#      boxes.append([x,y,w,h])
#      count+=1
#
#for box in boxes:
#    x, y, w, h = box
#    print(x,y,w,h)
##     imOut = imOut[x:x+w,y:y+h]
#    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
## plt.figure()
#plt.imshow(imOut)
#