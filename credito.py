from minisom import MiniSom
import pandas as pd 
import numpy as np
import os

# change current working directory
CURR_DIR = os.getcwd()

os.chdir(CURR_DIR)
# current working directory

# setting the actual csv file to be compilated
base = pd.read_csv("credit_data.csv")
base = base.dropna()
base.loc[base.age < 0 , "age"] = 40.92

# slicing result values from data
X = base.iloc[:, 0:4].values
Y = base.iloc[:,4].values

# data normalization to minimize process stress
from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler(feature_range = (0,1))
X =normalizador.fit_transform(X)

# creating the map from here :
som = MiniSom(x=15, y=15, input_len=4, random_seed=0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 1000)

# creating module visualization
from pylab import pcolor, colorbar, plot  

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

#marking possible suspects within the map record
for i,x in enumerate (X):
    w = som.winner(x)
    plot(w[0]+ 0.5, w[1]+0.5, markers[Y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[Y[i]], 
         markeredgewidth=0.3)    

#manually selecting on the map neurons that had strong tendencies to suspects
mapping = som.win_map(X)
suspects = np.concatenate((mapping[(5,9)],mapping[(6,9)], mapping[(6,10)],
                           mapping[(7,12)], mapping[(7,13)], mapping[(14,4)],
                           mapping[(14,6)], mapping[(13,11)],
                           mapping[(13,14)], mapping[(4,8)]), axis = 0)

# undo normalization
suspects = normalizador.inverse_transform(suspects)

classe = []

for i in range(len(base)) :
    for j in range(len(suspects)) :
        if base.iloc[i,0] == int(round(suspects[j,0])):
            classe.append(base.iloc[i,4])
# transforming list to a numpy element
classe = np.asarray(classe)

# finding the suspects that got the loan
suspects_end = np.column_stack((suspects, classe))

# finally here are all the suspects listed by order
# of those who are suspects and have been accepted, even those who are suspects but have not had accepted
suspects_end = suspects_end[suspects_end[:, 4].argsort()]

import time
sus = 1
for i in suspects_end:   
    print('\n')
    print("-"*100)       
    print(f'Suspect {sus} \n') 
    sus+=1
    for j in i :
        print(str(j.tolist())+"|",end='')
        time.sleep(0.1) 