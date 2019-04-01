#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data read , the csv files should be in the current directory 

import numpy as np
import pandas as pd 

input_data={}

for i in range(1,10):
    input_data[i]=pd.read_csv("transaction_20170"+str(i)+".csv")
for i in range(10,13):
    input_data[i]=pd.read_csv("transaction_2017"+str(i)+".csv")
    
for i in range(1,10):
    input_data[i+12]=pd.read_csv("transaction_20180"+str(i)+".csv")
for i in range(10,13):
    input_data[i+12]=pd.read_csv("transaction_2018"+str(i)+".csv")


# In[ ]:


# Preprocessing the Data and converting it to a matrix of vectors
# Caution : Do not run this code it will take some time , the arr matrix is already computed and stored as a pickle file so better 
# run the next block of code

arr=[]

zero=[]

for i in range(0,39):
    zero.append(0)

t=[]
for i in range(0,24):
    t.append([])
    

for i in range(0,40001):
    t=[]
    for i in range(0,24):
        t.append(zero)
    arr.append(t)

print(len(arr[0]), len(arr[0][0]))



    



# print(len(arr), len(arr[0]))

# print(input_data[1].loc[5,:])    
# print("hi")
for k in input_data.keys():
    no_of_val=input_data[k].shape[0]
    print(no_of_val)
    for serial in range(0,no_of_val) :
        temp=[]
        id_no=input_data[k].loc[serial,"id"]
        id_no = int(id_no)
        k = int(k)
#         print(type(int(id_no)), type(k))
        for col in input_data[k].columns:
            if col=="id":
                #id_no=
                continue
            else:
                temp.append(input_data[k].loc[serial,col])
#         print(id_no)
        #print(id_no, k-1)
            
        arr[id_no][k-1] = temp
        #print(arr[1][0])


# In[ ]:


import pickle

pickle_out = open("temp.pickle","rb")
arr = pickle.load(pickle_out)

pickle_out.close()


# In[ ]:


# Creating the clusters
from sklearn.cluster import KMeans


#print(len(arr[0]))
from sklearn.cluster import KMeans


X=[]
cluster=[]
for j in range (0,len(arr[0])):
    temp=[]
    for i in range(0,len(arr)):
        if not arr[i][j]:
            continue
        else:
            temp.append(arr[i][j])

    temp = np.array(temp)
    kmeans = KMeans(n_clusters=50, random_state=0).fit(temp)
    cluster=list(kmeans.labels_)
    X.append(cluster)



# print(arr[1][0])


# In[ ]:


#Finding dissimilarity between clusters
# Dont run this rather use the pickle code in the next block to get sim_data

sim_data=np.zeros(shape=[40000,23])

import difflib


for i in range(0,23):
    j=0
    hello=[]
    print(i)
    print("hellllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllo")
    visited1=np.zeros(shape=[40000,1])
    visited2=np.zeros(shape=[40000,1])
    while(j<40000):
#         print(j)
        if(visited1[j][0]==1 and visited2[j][0]==1):
            j=j+1
            continue
        temp=X[i]
        pos_1=[]
        for k in range(0,40000):
            if(X[i][j]==temp[k]):
#                 print(k)
                pos_1.append(k)
                visited1[k][0]=1
            
        temp=X[i+1]
        pos_2=[]
        for k in range(0,40000):
            if(X[i][j]==temp[k]):
#                 print(k)
                pos_2.append(k)
                visited2[k][0]=1
        
        #compare pos1 and pos2
        sm=difflib.SequenceMatcher(None,pos_1,pos_2)
        value=sm.ratio()
        for q in pos_1:
            sim_data[q][i]=value
            
        for q in pos_2:
            sim_data[q][i]=value
        
        sim_data[j][i]=value
            
        
        j=j+1
    
        
        


# In[ ]:


pickle_out = open("sim_data.pickle","wb")
pickle.dump(sim_data, pickle_out)
pickle_out.close()


# In[ ]:




tot_sum=0.0
for i in range(40000):
    for j in range(23):
        
        tot_sum=tot_sum+sim_data[i,j]

        
avg=tot_sum/(40000*23)

print(avg)



    


# In[ ]:


months=['Jan-17','Feb-17','Mar-17','Apr-17','May-17','Jun-17','Jul-17','Aug-17','Sep-17','Oct-17','Nov-17','Dec-17','Jan-18','Feb-18','Mar-18','Apr-18','May-18','Jun-18','Jul-18','Aug-18','Sep-18','Oct-18','Nov-18','Dec-18']

threshold=0.0
counter=0


p2m = open("peer2me.txt","w")
graph=[]
for i in range(1,40000):
    temp=[]
    for j in range(23):
        if(sim_data[i,j]>avg+threshold):
            counter=counter+1
            temp.append(1)
            p2m.write("User id : "+str(i) + " :: " + months[j]+" to "+ months[j+1]+" :: Change\n")
            print("User id  : "+str(i) + " :: " + months[j]+" to "+ months[j+1]+" :: Change\n")
        else:
            temp.append(0)
            p2m.write("User id : "+str(i) + ":: " + months[j]+" to "+ months[j+1]+" :: No Change\n")
            print("User id : "+str(i) + " :: " + months[j]+" to "+ months[j+1]+" :: No Change\n")
    
    graph.append(temp)

p2m.close() 


print(counter/23)


# In[ ]:


import matplotlib.pyplot as plt 

y_ = graph[34240]
x_ = list (range(len(months[0:-1])))

# plotting the points  
plt.plot(x_, y_, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=6) 
  
# naming the x axis 
plt.xlabel('Months') 
# naming the y axis 
plt.ylabel('Change') 
  
# giving a title to my graph 
plt.title('Behaviour Graph for user '+ str(34240)) 
  
# function to show the plot 
plt.show() 


# In[ ]:


import matplotlib.pyplot as plt 

y_ = graph[1000]
x_ = list (range(len(months[0:-1])))

# plotting the points  
plt.plot(x_, y_, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=6) 
  
# naming the x axis 
plt.xlabel('Months') 
# naming the y axis 
plt.ylabel('Change') 
  
# giving a title to my graph 
plt.title('Behaviour Graph for user '+ str(1000))
  
# function to show the plot 
plt.show() 


# In[ ]:


# Histogram

import seaborn as sns


hist_y=[]
hist_x=range(40000)
for i in range(0,39999):
    hist_y.append(sum(graph[i]))
    




plt.hist(hist_y)
plt.title("Frequency of no of changes of user behaviour in 24 months")
plt.xlabel("No of Behaviour changes")
plt.ylabel("Frequency")


    


# In[ ]:


#Behaviour Detection in small pockets 
#For Behaviour detection in small pockets we set the threshold value high 


months=['Jan-17','Feb-17','Mar-17','Apr-17','May-17','Jun-17','Jul-17','Aug-17','Sep-17','Oct-17','Nov-17','Dec-17','Jan-18','Feb-18','Mar-18','Apr-18','May-18','Jun-18','Jul-18','Aug-18','Sep-18','Oct-18','Nov-18','Dec-18']

threshold=-0.2
counter=0



p2m = open("peer2me_small.txt","w")
graph=[]
for i in range(1,40000):
    temp=[]
    for j in range(23):
        if(sim_data[i,j]>avg+threshold):
            counter=counter+1
            temp.append(1)
            p2m.write("User id : "+str(i) + " :: " + months[j]+" to "+ months[j+1]+" :: Change\n")
        else:
            temp.append(0)
            p2m.write("User id : "+str(i) + ":: " + months[j]+" to "+ months[j+1]+" :: No Change\n")
    
    graph.append(temp)

p2m.close() 
print(counter/23)



import matplotlib.pyplot as plt 

y_ = graph[2200]
x_ = list (range(len(months[0:-1])))

# plotting the points  
plt.plot(x_, y_, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=6) 
  
# naming the x axis 
plt.xlabel('Months') 
# naming the y axis 
plt.ylabel('Change') 
  
# giving a title to my graph 
plt.title('Behaviour Graph for user '+ str(1000))
  
# function to show the plot
plt.show() 

# Histogram

import seaborn as sns


hist_y=[]
hist_x=range(40000)
for i in range(0,39999):
    hist_y.append(sum(graph[i]))
    




plt.hist(hist_y)
plt.title("Frequency of no of changes of user behaviour in 24 months")
plt.xlabel("No of Behaviour changes")
plt.ylabel("Frequency")


    


# In[ ]:


#Another approach for pocket detection

from numpy import dot

def intersection(list1, list2): 
    return list(set(list1) & set(list2)) 

def cosine_similarity(a,b):
    if (dot(a,a) **.5)*(dot(b,b) ** .5)==0:
        return 0
    if (dot(a,b)*1.0/( (dot(a,a) **.5) * (dot(b,b) ** .5))>0.99):
        return 1#dot(a,b)*1.0/( (dot(a,a) **.5) * (dot(b,b) ** .5))
    else:
        return 0


All_cos=[]

for i in range(23):
    curr_clus=X[i]
    temp1=[]
    for k in range(len(curr_clus)):
        if(curr_clus[k]==curr_clus[1]):
              temp1.append(k)
    
    
#     cos_sim=[]
#     for q in temp:
#         cos_sim.append(cosine_similarity(list(arr[q][i]),list(arr[1][i])))
    
    curr_clus=X[i+1]
    temp2=[]
    for k in range(len(curr_clus)):
        if(curr_clus[k]==curr_clus[1]):
              temp2.append(k)
    
    r=intersection(temp1,temp2)

    if( len(r) < 0.75*(min(len(temp1),len(temp2)))):
        print("User no : "+str(i) + " :: " + months[i]+" to "+ months[i+1]+" :: Change\n")
    else:
        print("User no : "+str(i) + " :: " + months[i]+" to "+ months[i+1]+" :: No Change\n")
                
    

