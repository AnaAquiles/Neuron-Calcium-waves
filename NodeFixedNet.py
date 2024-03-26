#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:06:10 2022

@author: aaquiles
"""
### NODES FIXED NETWORKS

import sys, importlib, os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from scipy import stats 
import statsmodels.api as sm

"""
1)         FDR correction
"""

pvals = np.loadtxt('11CPilo-KClhPiloPVAL4-PEAK.csv', delimiter= ',')       #ENTER THE FILENAME OF PVALS
spcorr = np.loadtxt('11CPilo-KClhPiloSPCORR4-PEAK.csv', delimiter= ',')

#Modify the structure of the oval mat, to a trigualte and take the len of each axis
pval = np.tril(pvals)
I = len(pval[:,0])
#Vectorize the matrix and clean the zero values, 
pval1d = np.ravel(pval) 
pval1dNonZero = np.array(np.where(pval1d != 0))
pval1dZero = np.array(np.where(pval1d == 0))

#Calculate the FDR of the pval vectorizated
pval_matFDR = pval1d[pval1d != 0]
Fdr = sm.stats.fdrcorrection(pval_matFDR)
Fdr = np.asarray(Fdr)

#Take the binary result of FDR to localize the true pvals that will keep in
binaFDR = Fdr[0,:]
OneBin = np.array(np.where(binaFDR == 1))
ZeroBin = np.array(np.where(binaFDR == 0))
indFDR = np.take(pval1dNonZero,ZeroBin)

Zeross = np.zeros(len(indFDR[0,:]))
pvalZeroCorr = np.vstack((Zeross,indFDR)).T

#Build and array of zeros equal to the lenght of the zeros in my first pval vector
Zeros = np.zeros(len(pval1dZero[0,:]))
pval1Zero = np.vstack((Zeros,pval1dZero[0,:])).T

Pval_Zero = np.concatenate((pval1Zero,pvalZeroCorr))


indP_FDR = np.take(pval1dNonZero,OneBin)
indP_FDR = indP_FDR[0,:]
pval_corr =np.take(pval_matFDR,OneBin, axis=0)
Pval_corrInd = np.vstack((pval_corr[0,:],indP_FDR))

Pval_Zero = Pval_Zero.T

#Rebuild the matrix by the index of those pvals corrected and those zeros
Pvals_merged = np.hstack((Pval_Zero,Pval_corrInd)).T
ind = np.argsort(Pvals_merged[:,1],axis =0)
x = np.take(Pvals_merged[:,0],ind)
Pvals_sortRe = np.reshape(x, (I,I))


#Make the relation by the cells interactions
pvaltril=np.tril(Pvals_sortRe)
p = np.where((pvaltril<0.05) & (pvaltril!=0)) 
pp=[]
for i1,i2 in zip(p[0],p[1]):
    pp.append((i1,i2, pvaltril[i1,i2])) 
pp =np.array(pp)  

spcorr2=np.tril(spcorr)
z = np.where((spcorr2!=1) & (spcorr2!=0)) 
zz=[]
for i1,i2 in zip(z[0],z[1]):
    zz.append((i1,i2, spcorr2[i1,i2])) 
zz=np.array(zz)  


Pvals_sortReBool = Pvals_sortRe.astype(bool)*1
spcorrBool = np.tril(spcorr).astype(bool)*1
Intersect_mat  = Pvals_sortReBool * spcorrBool
matSpcorr = spcorr2 * Intersect_mat

matSpcorr_Df = pd.DataFrame(matSpcorr)
# matSpcorr_Df = matSpcorr_Df[IndexOrdered_Power]

i = np.where((matSpcorr != 0) & (matSpcorr > 0.1)) 
ii =[]

for i1,i2 in zip(i[0],i[1]):
    ii.append((i1,i2, matSpcorr[i1,i2])) 

ii=np.array(ii)

#%%

Weight = np.delete(ii,[1], axis = 1)  
Nodos_real = np.unique(Weight[:,0])   
                                          #number of nodes in my weight array
meanSpcorr = []
for i in range(0,len(Nodos_real)):
    meanSpcorr.append(np.mean(Weight[Weight[:,0]==i], axis = 0))

meanSpcorr = np.array(meanSpcorr)
meanScorr_nNa = pd.DataFrame(meanSpcorr)                                        #convert to DataFrame for make easier the nan deletion
meanScorr_nNa = meanScorr_nNa.dropna() 
NodeValue_R = meanScorr_nNa.values
NodeValue = NodeValue_R[:,0].astype(int)
rho = NodeValue_R[:,1]
i = len(NodeValue_R)



#%%
Node_RM = rm.sample(Nodes,nC[4])
M = np.take(ii,Node_RM, axis = 0)

## build a new graph
G= nx.Graph()

M_=np.delete(M,[2],axis=1)
MTotal_list = M_.tolist()
       
G.add_edges_from(MTotal_list) #conexion
G.add_weighted_edges_from(M) #pesos

Nnodos = nx.number_of_nodes(G)
Density = nx.density(G)
Cluster = nx.average_clustering(G)
Assortativity = nx.degree_assortativity_coefficient(G)
Degree_mixing = nx.degree_mixing_matrix(G)
Degree_dict = nx.degree_mixing_dict(G)

degrees = [G.degree(n) for n in G.nodes()]
degrees_Arr = np.array(degrees)

#%%

import random as rm

# Nodes = NodeValue.tolist()
#NodeBoundarie
# nB = np.array([584,629,613,555,529]) # bcnu window
# nC = np.array([353,365,361,363,365]) #control window
Cells = np.arange(0,len(spcorr2),1).tolist()

# Node_RM = rm.sample(Cells,300)
# M = np.take(matSpcorr,Node_RM,axis=0)
# M = np.takeM,Node_RM,axis=1)

# i = np.where(M !=0)
# ii =[]

# for i1,i2 in zip(i[0],i[1]):
#     ii.append((i1,i2, M[i1,i2])) 
# ii=np.array(ii)
    
def SurrogateNetNodeFixed(Node, N=100):  # WITH K, FOR STROGATZ
    # Graph = []
    Cluster = []
    Assort = []
    Eff = []
    Nodes =[]
    for i in range(N):
        # Node_RM = rm.sample(Node,300)
        M = np.take(matSpcorr,Node_RM,axis=0)
        M = np.take(M,Node_RM,axis=1)
        i = np.where(M !=0)
        ii =[]
        for i1,i2 in zip(i[0],i[1]):
            ii.append((i1,i2, M[i1,i2])) 
        ii=np.array(ii)
        G = nx.Graph() #Change to     watts_strogatz_graph   erdos_renyi_graph
        M_= np.delete(ii,[2],axis=1)
        MTotal_list = M_.tolist()
        G.add_edges_from(MTotal_list) #conexion
        G.add_weighted_edges_from(ii) #pesos
        ClusterRfast = nx.average_clustering(G) 
        AssortativityRfast = nx.degree_assortativity_coefficient(G)
        Effic_G = nx.efficiency_measures.global_efficiency(G)
        Nodes.append(Node_RM)
        Cluster.append(ClusterRfast)
        Assort.append(AssortativityRfast)
        Eff.append(Effic_G)
    NodesR = np.array(Nodes)
    ClusterSurr= np.array(Cluster)    
    AssortSurr= np.array(Assort)
    Efficiency = np.array(Eff)
    return ClusterSurr,AssortSurr,Efficiency, NodesR

ClusterSurr,AssorSurr, Efficiency, Node_RM = SurrogateNetNodeFixed(Cells)

ClustMean  = np.mean(ClusterSurr)
AssoMean = np.mean(AssorSurr)


#%%

df = pd.read_csv('BothnodeFixed2022IT.csv')

flatui = ["lightcoral","firebrick"]


sns.catplot(x="Group", y="Value",col = 'Metric', hue = 'Time'
            , palette="flare",height=6, aspect=.75,  #YlGnBu_d
            kind="box",
            data=df)
plt.ylim(-0.2,0.8)
plt.box(False)


#%%
df = pd.read_csv('MetricsControl-Peak2022.csv')

g = sns.catplot(x=df["Metric"], y=df["Value"], hue=df["x"], col=df["Time"],
                capsize=.2, palette="flare", height=6, aspect=.75,
                kind="point", data=df)

#%%


"""
      Distances relation to spcorr
"""

######## K

G= nx.Graph()

ii_=np.delete(ii,[2],axis=1)
zzTotal_list = ii_.tolist()

G.add_edges_from(zzTotal_list) #conexion
G.add_weighted_edges_from(ii) #pesos

degrees = [G.degree(n) for n in G.nodes()]
degrees_Arr = np.array(degrees)

# K_Total = np.take(degrees_Arr,WeightVaL[:-4])

######

Dist =pd.read_csv("DepthControlAll.csv")

SubjDist = Dist[(Dist['s']=='09A')]
DistVal = SubjDist['Depth'].values
DistVal_selected = np.take(DistVal,WeightVaL)

#%%
"""
   Folder DatosGraficas/Network/Peak
"""

df = pd.read_csv('Ctrlk-r-depth2022-01.csv')
datas = df[(df['x']=='10C')]

sns.relplot(
    data=df, x=df['K'], y=df['Depth'],
    col=df['Time'], hue=df['K'], size=df['K'],
    kind="scatter"
)


#%%

"""
   RHO VS TIME PER ANIMAL AND GROUP
   Folder /DatosGraficas
"""
df = pd.read_csv('Ctrlk-r-depth2022-01.csv')
sns.lineplot(data=Rho, x="Time", y="Rho")


#%%

"""
   RHO TOTAL HISTOGRAM 
"""

BCNU = pd.read_csv('RhoTOTAL_Control2022.csv')

da = BCNU[BCNU['x'] == '10C']


plt.figure(2)
sns.displot(data=BCNU, x="W0_K", 
            hue="Time", col="x", kind="kde", palette="rocket")
plt.box(False)
# plt.ylim(0,0.015)

plt.figure(4)
sns.histplot(data=da, x="W0_K", hue="Time", multiple="stack",
             palette="rocket",stat = 'probability')
plt.box(False)
plt.title('10C')

plt.xlim(-0.6,0.8)

#%%

import plotly.graph_objects as go

df = pd.read_csv('MetricsControl-Peak2022.csv')

dat = df[df['Time'] == 'Before']
Dats = dat[dat['Metric'] == 'Efficiency']
x = Dats['Value'].values 


fig = go.Figure()
fig.add_trace(go.Box(
    y = x,
    name="All Points",
    jitter=0.3,
    pointpos=-1.8,
    boxpoints='all', # represent all points
    marker_color='rgb(7,40,89)',
    line_color='rgb(7,40,89)'
))



fig.show()

plotly.offline.plot(fig) 
