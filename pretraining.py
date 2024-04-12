#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2#,keras
import constants
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
import sys
from PIL import Image
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
path = constants.path
annot = constants.annot
# Load all batches of data
X_batches = [np.load(f"preprocessing_results/X_new_{i}.npy") for i in range(0, 1)]
Y_batches = [np.load(f"preprocessing_results/Y_new_{i}.npy") for i in range(0, 1)]

# Concatenate the batches along the first axis
X_new = np.concatenate(X_batches, axis=0)
Y_new = np.concatenate(Y_batches, axis=0)

num_images = 25
count = 0
one_labels = []
for i in range(len(Y_new)):
    if Y_new[i] == 0:
        one_labels.append(i)
#pytorch 
transform = transforms.Compose([
    transforms.ToTensor(),
    # Transpose the image to (channels, height, width) before normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

custom_dataset = CustomDataset(X_new,Y_new,transform = transform)
train_loader = DataLoader(custom_dataset,batch_size=200,shuffle = True)
for images,labels in train_loader:
    print(images.shape)
    image = images[20].permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.show()
    break
# Define MobileNetV3 model
model = models.mobilenet_v3_small(pretrained=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#mobilenetv3.train()

num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(),'mobilenetv3_torch.pth')
#num_epochs = 4
#for epoch in range(num_epochs):
#    running_loss = 0.0
#    for inputs, labels in train_loader:
#        optimizer.zero_grad()
#        outputs = mobilenetv3(inputs)
#        loss = criterion(outputs,labels)
#        loss.backward()
#        optimizer.step()
#        running_loss +=loss.item() * inputs.size(0)
#    epoch_loss = running_loss/len(custom_dataset)
#    print(f"Epoch {epoch+1}/{num_epochs} LOSS:{epoch_loss}")
#
#
#torch.save(mobilenetv3.state_dict(),'mobilenetv3_torch.pth')
#






# Display images and corresponding labels
#fig, axes = plt.subplots(5, 5, figsize=(15, 9))  # Adjust the size of the figure as needed
#
#for i in range(num_images):
#    ax = axes[i // 5, i % 5]
#    ax.imshow(X_new[one_labels[i]])
#    ax.set_title(f"Label: {one_labels[i]}")
#    ax.axis('off')
#
#plt.tight_layout()
#plt.show()
#base_model = tf.keras.applications.MobileNetV2(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
#base_model.trainable = False
#x = base_model.output
#x = tf.keras.layers.GlobalAveragePooling2D()(x)
#predictions = tf.keras.layers.Dense(1,activation = 'sigmoid')(x)
#
#model = tf.keras.models.Model(inputs = base_model.input,outputs = predictions)
#print(base_model.input)
#model.summary()
#model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
#model.fit(X_new,Y_new,batch_size = 64,epochs = 3,verbose = 0,validation_split= 0.05,shuffle = True)
#model.save("my_model.h5")
##################################
#svm modl

#adding svm to last layer
#x =model.get_layer('fc2').output
#Y = tf.keras.layers.Dense(2)(x)
#final_model = tf.keras.Model(model.input,Y)
#final_model.compile(loss='hinge',
#              optimizer='adam',
#              metrics=['accuracy'])
#final_model.summary()
#final_model.load_weights('my_model_weights.h5')


#hist_final = final_model.fit(np.array(X_new),np.array(Y_new),batch_size=32,epochs = 20,verbose = 1,shuffle = True,validation_split = 0.05)



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