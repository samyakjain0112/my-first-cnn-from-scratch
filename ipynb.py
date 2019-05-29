#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
import tensorflow
import keras
import math


# In[4]:


#assumed that a_train is the set of all images in the form of numpy array of dimensions total_no_of_images*height_of_each_image*width_of_each_image*no.of_layers_of_each_image
#let us assume that the total no. of images are 50000
#so let batch_size=sqrt(5000) that is around 225
#so number of times we need to give in input for one epoch is 222
#let the number of epochs be 20
#let the image be 32*32*3
batch_size=225
n_epochs=20
alpha=0.05
#NOTE:
#WE CAN RUN THE WHOLE ALGORITHM 
#as alpha=np.pow(10,-np.random.randn(20))


# In[111]:


#FORWARD PROPOGATION
initial_size=0
for i in range(222):
    input_batch=a_data[initial_size:initial_size+batch_size]
    batch_layer_input=input_batch
    weight_layer_1=np.random.randint(0,20,(8,len(batch_layer_1[0]),5,5))
    batch_layer_input_padded=padding(batch_layer=batch_layer_input,size=1)
    unactivated_output_1=multiplication(batch_layer=batch_layer_input_padded,weight_layer=weight_layer_1,strides=1)
    activated_output_1=activation(unactivated_input=unactivated_output_1,activation='relu')
    batch_layer_pool_1=pooling(activated_input=activated_output_1,typ='max',strides=2,height=3,width=3)
    weight_layer_2=np.random.randint(0,20,(16,len(batch_layer_1[0]),5,5))
    batch_layer_padded_1=padding(batch_layer=batch_layer_pool_1,size=1)
    unactivated_output_2=multiplication(batch_layer=batch_layer_padded_1,weight_layer=weight_layer_2,strides=1)
    activated_output_2=activation(unactivated_input=unactivated_output_2,activation='relu')
    batch_layer_unroll=pooling(activated_input=activated_output_2,typ='max',strides=2,height=3,width=3)
    fully_connected_1=unroll(unroller=batch_layer_unroll)
    weight_fully_connected_1=np.random.randn(len(fully_connected_1[0]),40)
    fully_connected_2=activate(layer=multipy_fully_connected(layer=fully_connected_1,weights=weight_fully_connected_1),type='relu')
    weight_fully_connected_2=np.random.randn(len(fully_connected_2[0]),10)
    fully_connected_3=activate(layer=multipy_fully_connected(layer=fully_connected_2,weights=weight_fully_connected_2),type='relu')
    prediction=softmax(layer=fully_connected_3)


# In[112]:


#NOTE IN CASE WE OBSERVE GRADIENT VANISHING OR GRADIENT BOOSTING PROBLEM THEN WE CAN MULTIPLY THE WEIGHTS BY math.sqrt(2/no.of weights ie len(weights[0]) for fully connected and len(weights[0][0])*len(weights[0][0][0]) for convulational layer))
#the above was for relu activation but for sigmoid and tanh multiply by math.sqrt(1/no.of weights ie len(weights[0]) for fully connected and len(weights[0][0])*len(weights[0][0][0]) for convulational layer)


# In[32]:


print(np.random.randint(5,9,(8,3,8,7)))
print(np.random.randn(5,9))


# In[53]:


def padding(batch_layer,size):
    for i in batch_layer:
        for j in i:
            for _ in range(size):
                np.insert(j,0,np,zeros(len(j[0]),axis=1))
                np.insert(j,len(j),np,zeros(len(j[0]),axis=1))
                np.insert(j,0,np,zeros(len(j),axis=0))
                np.insert(j,(len(j)),zeros(len(j),axis=0))
    return batch_layer
            
            
    


# In[51]:


def multiplication(batch_layer,weight_layer,strides=1):
    unactivated_output=np.zeros([len(batch_layer),len(weight_layer),(len(batch_layer[0][0])-len(weight_layer[0][0]))/strides+1,(len(batch_layer[0][0])-len(weight_layer[0][0]))/strides+1])
    counter=-1
    filter_no=-1
    for j in batch_layer:
        counter+=1
        for i in weight_layer:
            filter_no+=1
            for initial_index_y in range(0,len(j[0])-len(i[0])+1,strides):
                for initial_index_x in range(0,len(j[0][0])-len(i[0][0])+1,strides):
                    for k in range(len(i)):
                        if initial_index_y+len(i[k])<=len(j[k]) & initial_index_x+len(i[k][0])<=len(j[k][0]):
                            temp=j[k][initial_index_y:len(i[k])+initial_index_y]
                            temp_final=[]
                            for i in temp:
                                temp_final.append(i[initial_index_x:len(i[k][0])+initial_index_x])
                            temp_final=np.array(temp_final)
                            unactivated_output[counter][filter_no][initial_index_y][initial_index_x]+=np.sum(np.dot(temp_final,i[k]))
    return unactivated_output
                    
                    
                    
    


# In[52]:


def activation(unactivated_input,activation='relu'):
    for i in range(len(unactivated_input)):
        for j in range(len(unactivated_input[i])):
            for k in range(len(unactivated_input[i][j])):
                for l in range(len(unactivated_input[i][j][k])):
                    if activation=='relu':
                        if unactivated_input[i][j][k][l]<0:
                            unactivated_input[i][j][k][l]=0
                    elif activation=='sigmoid':
                        unactivated_input[i][j][k][l]=sigmoid(unactivated_input[i][j][k][l])
                    elif activation=='tanh':
                        unactivated_input[i][j][k][l]=tanh(unactivated_input[i][j][k][l])
    return unactivated_input


# In[58]:


def pooling(activated_input,typ='max',strides=2,height=3,width=3):
    base_layer=np.zeros([len(activated_input),len(activated_input[0]),(len(activated_input[0][0])-height)/strides+1,(len(activated_input[0][0][0])-width)/strides+1])
    counter=-1
    for j in activated_input:
            counter+=1
            for initial_index_y in range(0,len(j[0])-height+1,strides):
                for initial_index_x in range(0,len(j[0][0])-width+1,strides):
                    for k in range(len(j)):
                        if initial_index_y+height<=len(j[k]) & initial_index_x+width<=len(j[k][0]):
                            temp=j[k][initial_index_y:height+initial_index_y]
                            temp_final=[]
                            for i in temp:
                                temp_final.append(i[initial_index_x:width+initial_index_x])
                            temp_final=np.array(temp_final)
                            if typ =='max':
                                base_layer[counter][k][initial_index_y][initial_index_x]=np.max(temp_final)
                            elif typ=='avg':
                                base_layer[counter][k][initial_index_y][initial_index_x]=np.mean(temp_final)
    return base_layer


# In[89]:


def unroll(unroller):
    p=0
    for i in unroller:
        a=[]
        for j in i:
            for k in j:
                for l in k:
                    a.append(l)
        if p==0:
            c=np.array(a)
            p=1
        else:
            np.insert(c,len(a),b,axis=1)
    return c


# In[87]:


#a=np.array([[1,2,3],[3,4,5],[6,7,8]])
#b=np.array([[4,6,3],[9,4,2],[8,1,7]])

#
# IMPORTANT EXAMPLE FOR UNDERSTANDING
#print(np.max(a))
#in insert the insertion is done before the specified index so to insert at last write len(object)
#c=np.array([])
#w=np.array([1,2,4])
#print(np.insert(a,1,w,axis=0))
#a=np.insert(a,len(a),[1,2,4],axis=0)
#print(a)


# In[90]:


def multipy_fully_connected(layer,weights):
    return np.dot(layer,weights)


# In[96]:


def activate(layer,type='relu'):
    for i in range(len(layer)):
        for j in range(len(layer[0])):
            if type=='relu':
                if layer[i][j]<0:
                    layer[i][j]=0
            elif type=='sigmoid':
                layer[i][j]=sigmoid(layer[i][j])
            elif type=='tanh':
                layer[i][j]=tanh(layer[i][j])
    return layer


# In[109]:


def dropout(layer,percentage,value):
    for i in range(len(layer)):
       
        if value==1:#means drop out of convulational layer
            for h in range(len(layer[i])):
                b=unroll(layer[i][k])
                k=np.random.randint(0,len(b),(percentage*len(b)))
                for j in k:
#considering square matrix of weights            
                        hr=int(j/len(layer[i][h]))
                        d=j%len(layer[i][h])
                        layer[i][h][hr][d]=0
        else:
            k=np.random.randint(0,len(layer[i]),(percentage*len(layer[i])))
            for j in k:
                layer[i][j]=0
    return layer
    #also multiply all oyher values by 1(1-percentage/100) \
    #this i have not implemented in the code


# In[91]:


#a=np.array([[1,2,3],[3,4,5],[6,7,8]])
#print(np.mean(a))


# In[93]:


def sigmoid(data):
    return 1/(1+math.exp(data))


# In[95]:


def tanh(data):
    return (math.exp(data)**2-1)/(math.exp(data)**2+1)


# In[94]:


def softmax(layer):
    s=0
    for i in layer:
        s+=math.exp(i)
    for i in layer:
        a.append(math.exp(i)/s)
    return a


# In[98]:


#************************************************************************************************************
#******************************************************************************************************************************************
#COMPLETED THE FOREWARD PROPOGATION NOW ITS TIME FOR BACKPROPOGATION:


# In[117]:


#BACKPROPOGATION
#note the convention used we have used that minor_delta_fully_connected or minor_delta_connected is ("da"),
#and the major_delta_fully_connected or major_delta_connected is ("dz") and costing_fully_connected or costing_connected is ("dw") similarly for major_delta_connected_poollayer_2
#note here we have initiated backpropogation by assuming that the last activation function is either sigmoiod or softmax thats why we have direcly taken delta as y-a
error=a_train_output-predictions
major_delta_fully_connected_2=error
#r=np.sum(fully_connected_2,axis=0)
#r=r.reshape(len(r),1)
costing_fully_connected_2=np.zeros(np.shape(weight_fully_connected_2))
for i in range(len(fully_connected_2)):
    costing_fully_connected_2+=fully_connected_2[i].T*major_delta_fully_connected_2[i]
weight_fully_connected_2-=alpha*costing_fully_connected_2/(len(fully_connected_2))
minor_delta_fully_connected_1=major_delta_fully_connectes_2*weight_fully_connected_2.T
major_delta_fully_connected_1=minor_delta_fully_connected_1*derivate_activation(activation_used='relu',layer=fully_connected_2)
costing_fully_connected_1=np.zeros(np.shape(weight_fully_connected_1))
for i in range(len(fully_connected_1)):
    costing_fully_connected_1+=fully_connected_1[i].T*major_delta_fully_connected_1[i]
weight_fully_connected_1-=alpha*costing_fully_connected_1/(len(fully_connected_1))
minor_delta_fully_connected_0=major_delta_fully_connectes_1*weight_fully_connected_1.T
major_delta_fully_connected_0=minor_delta_fully_connected_0*derivate_activation(activation_used='relu',layer=fully_connected_1)
major_delta_connected_poollayer_2=make_fully_connected_into_conv(delta_terms=major_delta_fully_connected_0,layer=batch_layer_unroll)


# In[116]:


def derivative_activation(activation_used='relu',layer):
    if activation_used=='relu':
        return 1
    elif activation_used=='sigmoid':
        return layer*(1-layer)
    elif activationn_used=='tanh':
        return 1-layer*layer


# In[110]:


def make_fully_connected_into_conv(delta_terms,layer):
    p=1
    init=0
    initial_conv=0
    for i in range(len(layer)*len(layer[0])):
        a=[]
        init=initial_conv
        for initial_index_convert in range(init,init+len(layer[0][0])*len(layer[0][0][0]),len(layer[0][0][0])):
            initial_conv=initial_index_convert
            a.append([delta_terms[initial_index_convert:len(layer[0][0][0])+initial_index_convert]])
        if p==1:
            arr=np.array(a)
            p=0
        else:
            np.insert(arr,len(arr),a,axis=0)
    arr=arr.reshape(len(layer),len(layer[0]),len(layer[0][0]),len(layer[0][0][0]))
    return arr


# In[ ]:


#BACKPROPOGATION FOR CONVULATIONOAL AND POOLING LAYER
major_delta_connected_convlayer_2=delta_pool_to_conv(activated_output=activated_output_2,batch_layer=batch_layer_unroll,major_delta_connected_poollayer=major_delta_connected_poollayer_2)
costing_connected_2=cal_gradient_conv(layer_delta=major_delta_connected_convlayer_2,layer_false_weight=batch_layer_padded_1,weights=weight_layer_2)
minor_delta_connected_poollayer_1=cal_delta_conv(weights=weight_layer_2,layer_delta=major_delta_connected_convlayer_2)
major_delta_connected_poollayer_1=minor_delta_connected_poollayer_1*derivative_activation(activation_used='relu',layer=batch_layer_pool_1)
major_delta_connected_convlayer_1=delta_pool_to_conv(activated_output=activated_output_1,batch_layer_pool=batch_layer_pool_1,major_delta_connected_poollayer=major_delta_connected_poollayer_1)
costing_connected_1=cal_gradient_conv(layer_delta=major_delta_connected_convlayer_1,layer_false_weight=batch_layer_input_padded,weights=weight_layer_1)
weight_layer_1-=alpha*costing_connected_1/len(batch_layer_pool_1)
weight_layer_2-=alpha*costing_connected_2/len(batch_layer_unroll)


# In[113]:



def delta_pool_to_conv(activated_output,batch_layer,major_delta_connected_poollayer):
    dup_activated_output=np.zeros(np.shape(activated_output))
    for i in range(len(batch_layer)):
        for j in range(len(batch_layer[i])):
            for k in range(batch_layer[i][j]):
                for l in range(batch_layer[i][j][k]):
                    for q in range(activated_output[i][j]):
                        for w in range(activated_output[i][j][q]):
                            if activated_output[i][j][q][w]==batch_layer[i][j][k][l]:
                                dup_activated_output[i][j][q][w]=major_delta_connected_poollayer[i][j][k][l]
    return dup_activated_output


# In[ ]:


def cal_gradient_conv(layer_delta,layer_false_weight,weights) :
    gradient=np.zeros(len(weight),len(layer_false_weight[0]),len(weights[0][0]),len(weights[0][0][0]))
    t=len(layer_false_weight[0])
    s=len(layer_delta)
    y=len(layer_delta[0])#no. of filters
    k=0
    init=0
    counter=0
    for j in range(s):
        for p in range(y):
            init=k+1 
            for i in range(init,init+t):
                layer_delta=np.insert(layer_delta[j],i+1,layer_delta[j][i],axis=0)
                k=k+1
            gradient[counter%y]+=(multiplication(batch_layer=layer_false_weight[j],weight_layer=layer_delta[j][init:init+t],strides=1)
    #strids should be same as the time of forward propogation
            counter+=1
return gradient


# In[123]:


d=np.array([[[1,2],[3,4]],[[1,4],[5,6]]])
print(np.sum(d,axis=1))


# In[ ]:


def cal_delta_conv(weights,layer_delta):
    #rotate weight vector by 180 degrees clockwise:
    


# In[ ]:


#we can implement ada booost for better results
#here we also need to implement for b
#note we can also implement the batch norm


# In[ ]:





# In[ ]:





# In[ ]:





# In[106]:


a=np.array([[1,2,3],[3,4,5],[6,7,8]])
t=np.sum(a,axis=0)
r=t.reshape(len(t),1)
print(r.shape)
print(r)


# In[ ]:




