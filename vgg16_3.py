import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from skimage import io
from skimage import filters
from skimage import color
import cv2
import os
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
#from keras.applications.vgg19 import VGG19
#from keras.applications.vgg16 import preprocess_input

images=[]
b1=[]

for filename in os.listdir("F:\dandruff"):
        img = cv2.imread(os.path.join("F:\dandruff",filename))
        if img is not None:
           im = cv2.resize(img, (224, 224)).astype(np.float32)
           images.append(im)
for i in range(0,len(images)):
 
                s=images[i]
                
               # ig= cv2.cvtColor( s, cv2.COLOR_RGB2GRAY )
                ig = s
                #im_power_law_transformation = cv2.pow(ig,0.6)
                b1.append(ig) 
images1=[]          

for filename in os.listdir("F:\hypertrichosis1"):
        img = cv2.imread(os.path.join("F:\hypertrichosis1",filename))
        if img is not None:
           im = cv2.resize(img, (224, 224)).astype(np.float32)
           images1.append(im)
for i in range(0,len(images1)):
 
                s=images1[i]
                
                #ig= cv2.cvtColor( s, cv2.COLOR_RGB2GRAY )
                ig = s
                #im_power_law_transformation = cv2.pow(ig,0.6)
                b1.append(ig) 
             
images3=[]             
for filename in os.listdir("F:\Tinia1"):
        img = cv2.imread(os.path.join("F:\Tinia1",filename))
        if img is not None:
            im = cv2.resize(img, (224, 224)).astype(np.float32)              
            images3.append(im)   
for i in range(0,len(images3)):
 
                s=images3[i]
                
                #ig= cv2.cvtColor( s, cv2.COLOR_RGB2GRAY )
                ig = s
                #im_power_law_transformation = cv2.pow(ig,0.6)
                b1.append(ig)           
# --- Till here i have 179 hair lesion images whose matrix form are stored in the list b1. having normalised form
data_train=[]                
for i in range(0,179):
   if(i<=58):
    data_train.append(0)  
   if(i>=59)and(i<=118):
    data_train.append(1)

   if(i>=119)and(i<=178):
    data_train.append(2)  

#----here in list called data_train ihave label of 179 images in a serial wise order;

from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(b1, data_train, test_size=0.20)        
#X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#--- here spliiting randomly into xtrain and xtest;

#initialise vgg16

#model = VGG19(weights='imagenet', include_top=False,input_shape=(224,224,3))
#from keras.preprocessing import image
# initialise dense net 121

from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input
model = DenseNet169(weights='imagenet', include_top=False,input_shape=(224,224,3))

#initialise resnet
'''
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
model = ResNet50(weights='imagenet')
'''

# till here initialise your model which you want to be used

print("extracting features")
features_train=[] 
for i in X_train:
  
   x = np.expand_dims(i, axis=0)
   x = preprocess_input(x)  
   features_train.append(model.predict(x))


features_test=[] 
for i in X_test:
  x1 = np.expand_dims(i, axis=0)
  x1 = preprocess_input(x1)
  features_test.append(model.predict(x1)  ) 

f_train=[]
f_test=[]

for i in features_train:
    f_train.append(i.flatten())

f_train=np.array(f_train)

for i in features_test:
    f_test.append(i.flatten())
    
f_test=np.array(f_test)


from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly',degree=3)
                     
svclassifier.fit(f_train,y_train)  

y_pred = svclassifier.predict(f_test)

y_pred1=[]
for i in range(0,len(y_pred)):
    y_pred1.append(y_pred[i])


 


#from sklearn.metrics import accuracy_score
#accuracy_score(f_test, y_pred1, normalize=False)

score=0
for i in range(0,len(y_pred1)):
    if(y_pred1[i]==y_test[i]):
        score+=1
accuracy= float(score)/len(y_pred1)
print accuracy*100        

from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(y_test,y_pred1))
print(classification_report(y_test, y_pred1)) 










