from minisom import MiniSom
import pandas as pd 
import numpy as np

base = pd.read_csv("credit_data.csv")
base = base.dropna()
base.loc[base.age < 0 , "age"] = 40.92

X=base.iloc[:, 0:4].values
y= base.iloc [:,4].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X =normalizador.fit_transform(X)
#criando efetivamente o mapa
som = MiniSom(x=15 , y =15, input_len= 4, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot

pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
color = ["r", "g"]
#marcando possiveis suspeitos dentro do registro de mapa
for i, x in enumerate (X):
    w= som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
markerfacecolor = 'None', markersize = 10,
markeredgecolor = color[y[i]], markeredgewidth = 2)

#selecionando manualmente no mapa neuronios que possuiam fortes tendencias a suspeitos
mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(5,6)],mapeamento[(6,14)], 
                            mapeamento [(8,7)],mapeamento[(7,14)]), axis = 0)
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []
#concatenando dados da base com a nova base de possiveis suspeitos
for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j,0])):
            classe.append(base.iloc[i,4])
           
classe = np.asarray(classe)
         
suspeitos_final = np.column_stack((suspeitos, classe))
#aqui estão os suspeitos listados em ordem
#dos que são suspeitos e foram aceitos, até os que são suspeitos mas não tiveram emprestimos aceitos
suspeitos_final = suspeitos_final[suspeitos_final[:,4].argsort()]