#!/usr/bin/env python
# coding: utf-8

# In[109]:


# pickle file data preprocessed

import pickle

f = open("temp.pickle", "rb")
arr = pickle.load(f)


# In[29]:


# Kmeans algorithm applied on every user id

from sklearn.cluster import KMeans
import numpy as np
ans = []
print("hi")
for i in range(0,len(arr)):
    X = np.array(arr[i])
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    ans.append(kmeans.labels_.tolist())
    


# In[26]:


# storing data in pickle

import pickle
f = open("me2me.pickle", "wb")
pickle.dump(ans, f)
f.close()


# In[3]:


# predictiong result for behaviour

f = open("me2me_out.txt", 'w')
for i in range(1, len(ans)):
    for j in range(1, len(ans[0])):
        f.write("user id : "+str(i)+" ")
        if j%12==0:
            f.write("12/2017 - 01/2018 ")
            if ans[i][j-1]==ans[i][j]:
                f.write("no change\n")
            else:
                f.write("change\n")
        if j%12==1:
            if j//12==0:
                f.write("01/2017 - 02/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("01/2018 - 02/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==2:
            if j//12==0:
                f.write("02/2017 - 03/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("02/2018 - 03/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==3:
            if j//12==0:
                f.write("03/2017 - 04/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("03/2018 - 04/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==4:
            if j//12==0:
                f.write("04/2017 - 05/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("04/2018 - 05/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==5:
            if j//12==0:
                f.write("05/2017 - 06/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("05/2018 - 06/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==6:
            if j//12==0:
                f.write("06/2017 - 07/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("06/2018 - 07/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==7:
            if j//12==0:
                f.write("07/2017 - 08/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("07/2018 - 08/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==8:
            if j//12==0:
                f.write("08/2017 - 09/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("08/2018 - 09/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==9:
            if j//12==0:
                f.write("09/2017 - 10/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("09/2018 - 10/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==10:
            if j//12==0:
                f.write("10/2017 - 11/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("10/2018 - 11/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
        if j%12==11:
            if j//12==0:
                f.write("11/2017 - 12/2017 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            else:
                f.write("11/2018 - 12/2018 ")
                if ans[i][j-1]==ans[i][j]:
                    f.write("no change\n")
                else:
                    f.write("change\n")
            
f.close()


# In[101]:


# ploting result for some random user-id

import pickle
import random
import matplotlib.pyplot as plt 
ind = random.randint(0, 40000)

f = open("me2me.pickle", "rb")
t = pickle.load(f)

temp = []
for i in range(1, 24):
    if t[ind][i-1]==t[ind][i]:
        temp.append(0)
    else:
        temp.append(1)

print(ind, temp)
plt.xticks(list(range(1, 24)))
plt.plot(list(range(1, 24)), temp)
plt.xlabel('months 1-23') 
plt.ylabel('(change-1),(no change-0)') 
plt.title('user id : '+str(ind)) 
# plt.legend() 
plt.savefig('month_change.png')
plt.show()


# In[102]:


# ploting no of behaviour change vs frequency

import matplotlib.pyplot as plt 

temp = []
for ind in range(1, 40001):
    c = 0
    for i in range(1, 24):
        if t[ind][i-1]==t[ind][i]:
            continue
        else:
            c += 1
    temp.append(c)

plt.hist(temp)
plt.title("Frequency of no of changes of user behaviour in 24 months")
plt.xlabel("No of Behaviour changes")
plt.ylabel("Frequency")
plt.savefig('userid_no_of_changes.png')
plt.show()


# In[50]:


# intersection of 2 list

def intersection(list1, list2): 
    return list(set(list1) & set(list2)) 


# In[51]:


# union of 2 list

def union(list1, list2): 
    return list(set(list1) | set(list2)) 


# In[59]:


# jaccard coeff

def jcoeff(list1, list2):
    i = len(intersection(list1, list2))
    u = len(union(list1, list2))
    return 1.0*i/u


# In[110]:


import pandas as pd 
import numpy as np
from numpy import dot


# In[111]:


# cosine semilarity

def cosine_similarity(a,b):
    if (dot(a,a) **.5)*(dot(b,b) ** .5)==0:
        return 0
    return dot(a,b)*1.0/( (dot(a,a) **.5) * (dot(b,b) ** .5) )


# In[120]:


# graph based clustering

import networkx as nx
th1 = 0.00
th2 = 0.0002

G = nx.Graph()
G.add_nodes_from(range(0, 24))
for i in range(0, 24):
    for j in range(0, i):
        if cosine_similarity(arr[1][i], arr[1][j])>=th1:
            G.add_edge(j, i)

edge_bc = nx.edge_betweenness_centrality(G)
max_bc = -1
for i in edge_bc:
    if edge_bc[i]>max_bc:
        max_bc = edge_bc[i]
        max_i = i

print(edge_bc)
while (max_bc>=th2) and (len(list(G.edges()))>0):
#     print("hi 123")
    edge_bc = nx.edge_betweenness_centrality(G)
    max_bc = -1.0
    for i in edge_bc:
        if edge_bc[i]>max_bc:
            max_bc = edge_bc[i]
            max_i = i
    G.remove_edge(max_i[0], max_i[1])


# In[121]:


# printing clusters

cc = nx.connected_components(G)
for i in cc:
    print("Length : ", len(i))
    print(list(i))

