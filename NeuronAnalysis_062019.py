#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:50:55 2018

@author: aaquiles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import signal
from scipy import stats 
import pandas as pd
import seaborn as sns 


filename="09ABasal"
data=np.loadtxt(filename + ".csv",delimiter=',')


#coors = np.loadtxt('coordenadasD190617.csv', delimiter= ',') 


f=len(data)
ini = f//2

datos=np.array([data[i:i+ini] for i in range(0,f,ini)])

#datos=np.array([data[i:i+1000] for i in range(0,2000,1000)])
datos=np.swapaxes(datos,1,2)  

#Normalization respect the baseline: total of datos - min baseline / min baseline. We considere only the first 50 images of basal activity
def NormF(datos):
    baseline=np.amin(datos[:,:,:ini],-1)[:,:,None]          #Hasta dónde se vamos a tomar de la actividad basal; hasta el valor 25     
    return datos/baseline
#Correction of activity cells debleach with linear regress of the first 50 values for each condition
def detrend(datos,window=ini):#arreglo indicado para las 300 imágenes,con regresión líneal 
    x=np.arange(0,window)
    x = x[None,:]*np.ones((datos.shape[-2],1))
    x=np.ravel(x)
    slopes=[]
    intercepts=[]
    for dat in datos:
        y = np.ravel(dat[:,:window])
        slope,inter,_,_,_=stats.linregress(x,y)
        slopes.append(slope)
        intercepts.append(inter)
        #-1 is the axis of ROI's
    slopes=np.array(slopes)
    intercepts=np.array(intercepts)
    t=np.arange(0,datos.shape[-1])
    trends=np.array((intercepts)[:,None] + np.array(slopes)[:,None] * t[None,:])
    return datos - trends[:,None,:]
b,a = signal.bessel(3,0.1,btype='lowpass') #grado del filtrado 0.1
datosfilt=signal.filtfilt(b,a,datos,axis=-1)
datosNorm=detrend(NormF(datos))
datosNormFilt=(NormF(datosfilt)) #sin la funcion detrend ajusta mejor la señal
dt=0.3
time=np.arange(0,dt*datosNorm.shape[-1],dt)  # segundos

# c,d = signal.bessel(3,0.003,btype='lowpass') #grado del filtrado 0.1
#datosfilt1=signal.filtfilt(c,d,datos,axis=-1)
#datosNormFilt1=detrend(NormF(datosfilt1))

#%%

# CUM SIGNALS 

plt.style.use('seaborn-poster')

fig, ax = plt.subplots(figsize=(200,100))


for ini in range(0,datos.shape[1]):
    for j in range(len(datos)):
        plt.plot(time + max(time)*j,datosNormFilt[j,ini,:], c ='dimgrey')
        
        
ax.set_xlim(0,900)
ax.set_ylim(1,1.05)
ax.set_xlabel('Time (s)')
ax.set_ylabel('$\Delta$ F/Fmin (a.u)')


plt.grid(False)    

plt.show() 



#%%

#Raster Plot of total signals 

from matplotlib import pyplot 
 
series = [] 
 
for ini in range(0,datos.shape[1]): 
    for j in range(len(datos)): 
        series.append(datosNormFilt[j,ini,:]) 
 
 
series = np.array(series) 
series = pd.DataFrame(series) 
 
pyplot.matshow(series, interpolation=None, aspect ='auto', cmap='bone', vmin = 1, vmax = 1.05) 
pyplot.colorbar() 
 
 
pyplot.xlabel('Time acquisition') 
pyplot.ylabel('nCells') 
pyplot.yticks(np.arange(0,596,298)) 
pyplot.xticks(np.arange(3000,0,)) 
pyplot.show()

#%%

# Plot cell responses exAbcles 
plt.style.use('seaborn-whitegrid')


plt.subplot(221)
plt.plot (time, datosNormFilt[0,4,:], c = 'coral')
plt.xlim(0,900)
plt.ylim(1,1.05)
plt.ylabel('$\Delta$ F/Fmin(a.u)')
plt.axvspan(50,200, alpha = 0.3, color = 'lightcoral')
plt.annotate('Pilo 300 $\mu$M', xy=(2,1), xytext=(10,1.05),fontsize=12, fontweight='bold')
plt.grid(False)


plt.subplot(222)
plt.plot (time, datosNormFilt[1,4,:],c = 'cornflowerblue')
plt.xlim(0,900)
plt.ylim(1,1.05)
plt.axvspan(50,200, alpha = 0.3, color = 'khaki')
plt.annotate('KCl 140 mM', xy=(2,1), xytext=(10,1.05),fontsize=12, fontweight='bold')
plt.grid(False)


plt.subplot(223)
plt.plot (time, datosNormFilt[0,84,:], c = 'coral')
plt.xlim(0,900)
plt.ylim(1,1.05)
plt.xlabel('Time (s)') 
plt.ylabel('$\Delta$ F/Fmin(a.u)') 
plt.axvspan(50,200, alpha = 0.3, color = 'lightcoral')
plt.grid(False)
 

plt.subplot(224)
plt.plot (time, datosNormFilt[1,84,:],c = 'cornflowerblue')
plt.xlim(0,900)
plt.ylim(1,1.05)
plt.xlabel('Time (s)')
plt.axvspan(50,200, alpha = 0.3, color = 'khaki') 
plt.grid(False)


#%%%%%
#______________________________________________________________________________

# Frequency and Amplitud response 
####         Option 1

import scipy as sy
import scipy.fftpack as syfp



N = (datosNormFilt.shape[-1])

i= 0
Treatment= datosNormFilt[i,56,:]

#______________________________________________________________________________
#______________________________________________________________________________
#______________________________________________________________________________


fftabsNumpy = sy.fft(datosNormFilt,axis= -1)

fftabsPilonumpy = (fftabsNumpy[0,56,:])


time_seg = 0.3 

Freq = []

for i in range(0, len(fftabsPilonumpy)):
    Frequency = syfp.fftfreq(len(fftabsPilonumpy[i,:]),time_seg)
    Freq.append(Frequency)
#idx = np.argsort(freqs) 

Abclitude = sy.log10(abs(fftabsPilonumpy)) 

meanAbc = np.mean(Abclitude,axis = 0)
sdAbc = np.std(Abclitude,axis = 0) 

Abclitude[Abclitude<(meanAbc + 0.5*sdAbc)]=0


plt.subplot(211)
plt.plot(time, Treatment.T) 
plt.xlabel('Time') 
plt.ylabel('F/Fmin') 

plt.subplot(212) 
plt.plot(Frequency, Abclitude.T,'.')
plt.xlabel('Freq (Hz)') 
plt.ylabel('Amplitude (log10)')
#plt.axvline(x=0.2, ymin= -4.3, ymax= 1.3,c='r',)
plt.xlim(0,1.5)

plt.show()

#%%

# Spectogram for each frequency and amplitude values FOR INDIVIDUAL CELLS

sample_rate= 3.3 #ms


fftabsPilonumpy = (datosNormFilt[0,25,:]) 


fff,t,Sxx=signal.spectrogram(fftabsPilonumpy,sample_rate)

Sxx = np.mean (Sxx)



plt.figure(1)
plt.clf()

plt.subplot(313)
plt.pcolormesh(t,fff,Sxx, cmap = 'RdBu') 
plt.ylim(0,0.2,0.05)
plt.xlabel('Time(sec)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()

plt.subplot(311)
plt.plot(time, fftabsPilonumpy, c='maroon')
plt.xlabel('Time(sec)')
plt.xlim(1,900,100)
plt.ylabel('F/Fmin [au]')


#%%
##_____________________________________________________________________________
####                     Option 2
#
#------------------For individual cells ---------------------------------------

sample_rate= 3.3 #ms


fftabsPilonumpy = (datosNormFilt[0,80,:])
fourier = np.fft.fft(fftabsPilonumpy)

# Freq values for x axis

fx_step_size = (sample_rate/len(fftabsPilonumpy))
nyq = .5*sample_rate

total_steps = nyq/fx_step_size 
fx_bins = np.linspace(0,nyq,total_steps)

#Smooth method "Welch's method"

ff,pspec =  signal.welch(fftabsPilonumpy, fs = 3.3, window ='hanning',
                        nperseg=300, noverlap = 0.2, 
                        nfft= None, detrend = 'linear',return_onesided= True,
                        scaling='spectrum')


#Freq_Amplitude ID array

Freq_Amp= np.column_stack((ff,pspec)).astype(float)

try_1 = np.where(ff<0.25)
try_1 = np.array(try_1).T.astype(float)

# Freqs selection with a 0.25 Hz boundary 

Freq_opt = []
for i in range(0,len(try_1)):
    Freq_opt.append(Freq_Amp[int(try_1[i]),:])
    
Freq_opt = np.array(Freq_opt)

    
# Power discrimination following the mean and Standar Deviation of the global
#               data distribution 

Amplitude = np.array(pspec)

meanAmp = np.mean(Amplitude,axis = 0)
sdAmp = np.std(Amplitude,axis = 0) 

#Amplitude[Amplitude<(meanAmp + 0.5*sdAmp)]=0

Abc_Amp = simps(Amplitude) # Area Under the curve of the power spectra 

plt.figure(1)
plt.clf()

plt.subplot(411)
plt.plot(time, fftabsPilonumpy.T, c = 'steelblue')
plt.xlabel('Time (s)')
plt.ylabel('F/Fmin')

plt.subplot(412)
plt.plot(fx_bins[0:1500], abs(fourier[0:1500]), c = 'steelblue')
plt.ylabel('Power')
plt.xlabel('Frequency (Hz)')

plt.subplot(413)
plt.plot(fx_bins[0:1500],np.log(abs(fourier[0:1500])),c = 'steelblue')
plt.ylabel('log Power')
plt.xlabel('Frequency (Hz)')

plt.subplot(414)
plt.plot(Freq_opt[0], Freq_opt[1],c = 'steelblue' )
plt.ylabel('Power (w/Smooth)')
plt.xlabel('Frequency (Hz)')


plt.figure(2)
plt.clf()
plt.hist(pspec, bins=5) 

#%%

#                      Plot neurons train spikes 
#                           by ISI's time for one cell FIRST

"""
         Comparacion de dos celulas en el tiempo de cambio/ velocidad de cambio
"""

import peakutils 
import plotly.graph_objs as go
import plotly
from scipy.signal import argrelextrema


st2 = (datosNormFilt[0,4,:]) 
st3 = (datosNormFilt[0,84,:]) 



#CELL 1
indices = peakutils.indexes(st2, thres=0.3/max(st2), min_dist=0.1) 
Fluo_values = np.take(st2, (indices))
Fluo_steps = np.array([j-i for i,j in zip (Fluo_values[:-1], 
                                           Fluo_values[1:])])
    
steps = np.array([j-i for i,j in zip (indices[:-1], indices[1:])]) 
time_step = (dt*steps)                                                        # valores de entre picos en segundos 
rate_change = (Fluo_steps/time_step)

#medidas del cambio
min_val_Tstep = np.amin(time_step)
max_val_Tstep = np.amax(time_step)
var_Tstep = np.var(time_step)


#CELL2
indices1 = peakutils.indexes(st3, thres=0.3/max(st3), min_dist=0.1) 
Fluo_values1 = np.take(st3, (indices1))
Fluo_steps_1 = np.array([j-i for i,j in zip (Fluo_values1[:-1], 
                                             Fluo_values1[1:])])

steps_1 = np.array([j-i for i,j in zip (indices1[:-1], indices1[1:])])
time_step1 = (dt*steps_1)                                                       # valores de entre picos en segundos 
rate_change1 = (Fluo_steps_1/time_step1)


min_val_Tstep = np.amin(time_step1)
max_val_Tstep = np.amax(time_step1)
var_Tstep = np.var(time_step1)

#aceleration = (np.diff(Fluo_steps)/np.diff(time_step1))


    
    
#  Plot offline the la señal de una celula con los peaks trazados

trace = go.Scatter(
        x=[j for j in range(len(st2))],
        y=st2,
        mode='lines',
       
        name='Original Plot'
) 

trace2 = go.Scatter(
        x=indices,
        y=[st2[j] for j in indices],
        mode= 'markers',
        marker=dict(
                size=8,
                color='rgb(255,0,0)',
                symbol='cross')) 

dat =[trace,trace2] 

plotly.offline.plot(dat) 

#_____________________________________________________________________________


plt.figure(2)
plt.clf()

plt.subplot(321)
plt.plot(time, st2, 'r-',c = 'coral')
plt.xlim(0,900)
plt.ylim(1,1.02)
plt.xlabel('time (s)')
plt.ylabel('$\Delta$ F/Fmin(a.u)') 
plt.grid(False) 

plt.subplot(325)
plt.plot(time_step, 'ko')
plt.xlabel('time (s)')
plt.ylabel('ISI(ms)') 
plt.grid(False) 

plt.subplot(323)
plt.hist(rate_change, bins = 50, color = 'darkslategrey')
plt.axhline(0,xmin=0,xmax=85, c = 'k', alpha=0.2) 
plt.ylabel('v = ($\delta$F(a.u))/($\delta$t(s))') 
plt.grid(False) 

plt.subplot(322)
plt.plot(time, st3, c = 'coral')
plt.xlim(0,900)
plt.ylim(1,1.02)
plt.grid(False) 

plt.subplot(326)
plt.plot(time_step1, 'ko')
plt.xlabel('time (s)')
plt.grid(False) 

plt.subplot(324)
plt.hist(rate_change1, bins = 50, color = 'darkslategrey')
plt.axhline(0,xmin=0,xmax=85, c = 'k',alpha=0.2) 
plt.grid(False) 


#%%_---------------------GENERAL FORM-------------------------------------------
"""
                   Conteo de peaks y analisis de sincronia

"""
import math
 
St = (datosNormFilt[0,:,600:3000])


#coors= np.loadtxt('10C-CoorAll.csv', delimiter= ',') # colocar el nombre del archivo con la extensión.csv

inx=[]

for n in range(0,len(St)):
    inx.append(peakutils.indexes((St[n,:]), thres=0.3/max(St[0]),              #  Busca los frames en donde cda imagen tiene un peak 
                                 min_dist=0.1))                                #


df = pd.DataFrame(inx).T                                                       # Lo convertimos en un DataFrame de pandas

#Matriz de 0 y 1 de todos los eventos de mi registro

s = (298,2400)          #para df[nCells,tiempo],, para xlocdf[tiempo,nCells]                                                       #matriz con solo zeros
ss = np.zeros(s)


for i in range (0,len(df)): # renglones
    for j in range (0,df.shape[1]): # columnas
        k = df[j][i].astype(int)
        if not math.isnan(df[j][i]):
            ss[j][k] = 1 
            
# Plot a eventplot 
plt.figure(3)
plt.clf()
plt.imshow(ss,interpolation='nearest')

# Plot a non acumulate eventplot 
plt.figure()
plt.clf()
plt.plot(ss,'--').T
plt.ylim(0.9,0.91)

#¿cuantas celulas tienen un peak en la misma imagen? Sincronia 

sync = [] 
Loc = [] 

for i in range(2400): 
    sync.append(df.eq(i,axis='columns'))                                        # deja unicamente las imagenes en donde responden simultaneamente las cels, con True/False, cuando no
    
for j in range(len(sync)):
    Loc.append(np.where(sync[j] == True)) 

xLoc = (map(list, zip(*Loc)))
#Matriz de sincronia arrojando el # de ROI que responde en la misma imagen 
xLocdf = pd.DataFrame(xLoc[1]) 


#Matriz binaria con solo los valores de tiempo de sincronia 

for j in range (0,len(xLocdf)): # renglones
    for i in range (0,xLocdf.shape[1]): # columnas
        k = xLocdf[j][i].astype(int)
        if not math.isnan(xLocdf[j][i]):
            ss[i][k] = 1 

plt.matshow(ss, interpolation = 'nearest')

# cambiar nuestra matriz a valores  0/1    


for i in range(0,xLocdf.shape[1]): 
    for j in range(0,len(xLocdf)): 
        if xLocdf[i][j] > 0:
            xLocdf[i][j] = 1
        else: 
            xLocdf[i][j] = 0         

        
#  Intervalo de tiempo entre spigas ISI, 

x = df.diff()
Steps_x = (x*dt)                                                               # Steps_x = Steps_x.fillna(0) 
Steps_x = Steps_x.drop([0])

lenght = np.array(Steps_x.count()).astype(float)                               # Cuantos peaks por celula


                                                                               # df_steps.append([j-i for i,j in zip (df[x,:-1], df[x,1:])])
x = coors[:,0]
y = coors[:,1]

plt.figure(3)
plt.scatter(x,y, s = 20, c= lenght, cmap = "seismic",alpha =0.5) 
plt.colorbar()
plt.show()


tmp=xLocdf[0] 
#%% 

"""
              Connection Inference TRANSFER ENTROPY 
Theory explanaton : http://dx.doi.org/10.1016/j.cnsns.2016.12.008
Theory application to neuron artificial net: doi:10.1371/journal.pone.0027431

jPype python's library include all the functions necessaries for TE analysis. Is an interfaz of java and python

___________

1) time-delay reconstruction of a phase space where you have X = data, Xi, 
        where i = time
    how to choose the time delay of my data? 
    - By, mutual information, and autocorrelation 
2) measure the probability of one event happens: p1, p2..pN, where log1/p(i) = information quantity of 
        certain event

3) measure the entropy of one or two random variables starting with the probabilities  of ocurrence 

        
"""

import jpype 
import random
import numpy 

jarLocation = "../../infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Generate some random binary data.
sourceArray = [random.randint(0,1) for r in range(100)]
destArray = [0] + sourceArray[0:99]
sourceArray2 = [random.randint(0,1) for r in range(100)]

# Create a TE calculator and run it:
teCalcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
teCalc = teCalcClass(2,1)
teCalc.initialise()

# First use simple arrays of ints, which we can directly pass in:
teCalc.addObservations(sourceArray, destArray)
print("For copied source, result should be close to 1 bit : %.4f" % teCalc.computeAverageLocalOfObservations())
teCalc.initialise()
teCalc.addObservations(sourceArray2, destArray)
print("For random source, result should be close to 0 bits: %.4f" % teCalc.computeAverageLocalOfObservations())

# Next, demonstrate how to do this with a numpy array
teCalc.initialise()
# Create the numpy arrays:
sourceNumpy = numpy.array(sourceArray, dtype=numpy.int)
destNumpy = numpy.array(destArray, dtype=numpy.int)
# The above can be passed straight through to JIDT in python 2:
# teCalc.addObservations(sourceNumpy, destNumpy)
# But you need to do this in python 3:
sourceNumpyJArray = jpype.JArray(jpype.JInt, 1)(sourceNumpy.tolist())
destNumpyJArray = jpype.JArray(jpype.JInt, 1)(destNumpy.tolist())
teCalc.addObservations(sourceNumpyJArray, destNumpyJArray)
print("Using numpy array for copied source, result confirmed as: %.4f" % teCalc.computeAverageLocalOfObservations())

jpype.shutdownJVM() 


#%%
#incompleto!!!!


def rateofchange(string):
    Fluo_val = []
    indices =[]
    
    for st in range(0,len(string)):
        indices.append(peakutils.indexes((string[st,:]), 
                                 thres=0.3/max(string[st]), min_dist=0.1))
        for s in range(0,len(string)):
            for n in range(0,len(indices)):
                Fluo_val.append(np.take((string[s,:]),(indices[n])))
    return indices

global_rate = rateofchange(St)
#    FluoSteps = np.array([j-i for i,j in zip (Fluo_val[:,:-1],
#                                              Fluo_val[:,1:])])
#    TimeSteps = (np.array([j-i for i,j in zip (indices[:,:-1],
#                                               indices[:,1:])]))*dt
#    return FluoSteps/TimeSteps 

#%% 
"""
SAME FUNCTION WITH PANDAS and Scipy (minimun point for inhibitory action)

"""
#____________________________________________________________________________

df =pd.DataFrame(st2, columns =['data'])

df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal, order=n)[0]]
['data']
#df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal, order=n)[0]]
#['data']


plt.scatter(df.index,df['min'], c='y')
#plt.scatter(df.index,df['max'], c='b')
plt.plot(df.index, df['data'], 'k-') 
plt.show()
#%%

# Frequency and Amplitud response Option 2 
#______________________________________________________________________________
#______________________________________________________________________________
"""
####---------------Frequency and Amplitud GLOBAL analysis----------------------
_______________________________________________________________________________

"""
i = 0

fftabsdata = (datosNormFilt[i,:,:])
    
AbcTotal=[]
for n in range(0,len(fftabsdata)):
    f,pspec =  signal.welch(fftabsdata, fs = 3.3, window ='hanning',
                        nperseg= 300, noverlap = 3.3//2, 
                        nfft= None, detrend = 'linear',return_onesided= True,
                        scaling='spectrum')
    

mean_A = np.mean(pspec, axis = 0)
var_A= np.var(pspec, axis = 0)

# Frequency partition min = <0.05, max 0.21 [Hz]
    
Ff_ = np.array(np.where(f<0.05)).T.astype(float)
Ff_1 = np.array(np.where((f<0.10) & (f>0.05))).T.astype(float)
Ff_2 = np.array(np.where((f<0.15) & (f>0.10))).T.astype(float)
Ff_3 = np.array(np.where((f<0.21) & (f>0.15))).T.astype(float)

# <0.05 Hz

Freq_part = []
for i in range(0,len(Ff_3)):
    Freq_part.append(pspec[:,int(Ff_3[i])])
    
Freq_part = np.array(Freq_part)
meanFreq_part = np.mean(Freq_part,axis = 1)
sdFreq_part = np.std(Freq_part,axis = 1) 
varFreq_part = np.var(Freq_part,axis = 1)

AmpFreq_part = np.mean(Freq_part, axis=0)


#Amplitude = np.array(pspec)
#meanAmp = np.mean(Amplitude,axis = 0)
#sdAmp = np.std(Amplitude,axis = 0) 
#varAmp = np.var(Amplitude,axis = 0)
#    
##    Amplitude[Amplitude<(meanAmp + 0.5*sdAmp)]=0
#Amp = np.mean(Amplitude, axis=1)

plt.figure(4)
plt.clf()

plt.subplot(131)
plt.plot(f,pspec.T, c='k')
plt.plot(f,mean_A, c='r')
plt.plot(f,var_A, c= 'b')

plt.subplot(132)
plt.plot(Ff_3,Freq_part, c='b')
plt.plot(Ff_3,meanFreq_part, c='r')
plt.plot(Ff_3,varFreq_part, c='k')
#plt.xlim(0,0.05)
#plt.legend('Global PSD','mean Global PSD', 'SD Global PSD')
plt.ylabel('Power (w/Smooth)')
plt.xlabel('Frequency (Hz)')
plt.title('Global PowerSpectra')


Abc = simps(pspec)
meanAbc = np.mean(Abc)
sdAbc = np.std(Abc)

plt.subplot(133)
plt.hist(AmpFreq_part, bins=10, )
plt.ylabel('Frequency')
plt.xlabel('Power values')
plt.title('Global Power values')
    

#%%

# Scatter plot with each power value normalizated

sns.set(style="ticks")
#sns.set(font_scale = 1.20)


df = pd.read_csv("11BCoor_p4.csv")
ax1 = df.plot.scatter(x='Y',
                      y= 'X',
                      c = 'power',
                      alpha=.90,
                      colormap = 'viridis',
                      vmin=1.5e-7, vmax= 4.0e-7
                     )

#%%

"""
Deep position along motor cortex 
"""

# Coordenadas 


#if nargin < 2
#   plotLines = false; 
#end


#plot(xy(:,1),xy(:,2),'. k')
#fract = 0.2; % fraction of space to pad xlim and ylim.
#xlim = get(gca,'XLim'); # que hace gca
#ylim = get(gca,'YLim'); #
#dx = xlim(2) - xlim(1);
#dy = ylim(2) - ylim(1);
#newxlim = [xlim(1)-dx*fract xlim(2)+dx*fract];
#newylim = [ylim(1)-dy*fract ylim(2)+dy*fract];
#set(gca,'XLim',newxlim,'YLim',newylim);
#

#hold on
#
#fprintf(1,'\nPlease draw the line of the pial surface\n\n');
#h = imfreehand('Closed',false);
#lxy = h.getPosition;
#lx = lxy(:,1);
#ly = lxy(:,2);
#plot(lx,ly,' -k');
#nSegments = size(lx,1) -1;
#
#
#depths = nan(1,nPoints);
#n = progress('init','Calculating distances');
#for i = 1 : nPoints
#    n = progress(i/nPoints,n);
#    p = xy(i,:);
#    distances = nan(nSegments,1);
#    points = nan(2,nSegments);
#    for s = 1 : nSegments
#        line_pA = [lx(s);  ly(s)];
#        line_pB = [lx(s+1);ly(s+1)];
#        [pn, dist, prel] = segment_point_near_2d(line_pA, line_pB, p');
#        distances(s) = dist;
#        points(:,s) = pn;
#    end 
#    [m, idx] = min(distances);
#    pnf = points(:,idx);
#    if plotLines; plot([p(1);pnf(1)],[p(2);pnf(2)],'Color',[0.9 0.9 0.9]);end
#    depths(i) = m;
#end
#progress('close');
#
#scatter(xy(:,1),xy(:,2),30, depths, 'filled')
#colorbar
#
#% return axes to their original extent
#set(gca,'XLim',xlim,'YLim',ylim)














































#%%

#Attractor ODE example FUNCTIONS

#Runge -Kutta INTEGRATOR

from mpl_toolkits.mplot3d import Axes3D

def generate(data_length, odes, state, parameters):
    data = np.zeros([state.shape[0], data_length])

    for i in xrange(5000):
        state = rk4(odes, state, parameters)

    for i in xrange(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state

    return data

def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
def lorenz_odes((x, y, z), (sigma, beta, rho)):
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        np.array([-8.0, 8.0, 27.0]), np.array([10.0, 8/3.0, 28.0]))

#the Rössler equations:

def rossler_odes((x, y, z), (a, b, c)):
    return np.array([-y - z, x + a * y, b + z * (x - c)])


def rossler_generate(data_length):
    return generate(data_length, rossler_odes, \
        np.array([10.0, 0.0, 0.0]), np.array([0.15, 0.2, 10.0]))
    

plt.figure(5)
plt.clf()
data = lorenz_generate(2**13)
plt.plot(data[1])



plt.figure(6)
plt.clf()    
figure = plt.figure()
axes = Axes3D(figure)
axes.plot3D(data[0], data[1], data[2])
figure.add_axes(axes)
plt.show() 
#%%


# difference in time for a select cell and during stimulus


# Derivada del tiempo en cada celula


Cells= datosNormFilt[0,:,:] 

time2 = time[25:175]


dC = []

for i in range(0, len(Cells)):
    dC.append(np.diff(Cells[i,:]))
    dCell = np.array(dC)
    dtime = np.diff(time)

dT = np.diff(time,1)

plt.figure(2)
plt.clf()

plt.plot(time[:2999], dCell.T, '.', c='y')

# exponential fitting

Cell1 = datosNormFilt[0,319,:]     
  
Cell1_d = np.diff(Cell1)/np.diff(time) 


Tau = np.polyfit(time, np.log(Cell1),1)

plt.figure(3)
plt.clf()

plt.subplot(311)
plt.plot(time, Cell1, c = 'darkslategrey')
plt.ylabel(r'$\Delta$F/F0 (a.u)')
plt.xlabel('Time (s)')
plt.xlim(1,900)
plt.grid(False)


plt.subplot(312)
plt.plot( Cell1_d, '.', c='darkslategrey')
plt.ylabel(r'($\Delta$F/F0)/dt')
plt.xlabel('Time (s)')
plt.xlim(1,900)
plt.grid(False)

plt.subplot(313)
plt.plot(dT, Cell1_d, '.', c='darkslategrey')
plt.ylabel(r'($\Delta$F/F0)/dt')
plt.xlabel('dTime (s)')
plt.grid(False)

#%%


"""
THEORICAL TIME, INTERSPIKE INTERVAL 

"""
import numpy as np


class SpikeTrain(object):

    def __init__(self, spikes, duration):
        """
        Initialize a spike train
        :param spikes: array of times of occurrences of spikes, in seconds
        :param duration: duration of this train, in seconds
        """
        self.spikes = spikes
        self.duration = duration

    def spike_counts(self, interval):
        """
        Produces the spike counts array of this spike train, given the interval
        :param interval: counting interval, in milliseconds
        :return: an array of spike counts
        """
        interval_sec = interval / 1000.
        return np.diff([np.count_nonzero(self.spikes < t) for t in np.arange(0, self.duration, interval_sec)])

    def interspike_intervals(self):
        """
        Returns the interspike intervals of this spike train
        :return: numpy array
        """
        return np.diff(self.spikes)

    def coefficient_variation(self):
        """
        The Coefficient of variation C_v = sigma_tau / mean_tau
        :return:
        """
        interspike = self.interspike_intervals()
        return np.std(interspike) / np.mean(interspike)

    def fano_factor(self, counting_intervals):
        """
        Compute the Fano factor sigma^2_n / mean_n with every given counting interval
        :param counting_intervals:
        :return:
        """
        ls = []
        for i in counting_intervals:
            counts = self.spike_counts(i)
            ls.append(np.var(counts) / np.mean(counts))
        return np.asarray(ls)

    def interspike_interval_histogram(self, bins):
        """
        Compute the interspike interval histogram: number of intervals falling in discrete time bins.
        :param bins: margins of the bins, in milliseconds
        :return: numpy array
        """
        intervals = self.interspike_intervals()
        return np.diff([np.count_nonzero(intervals < (t / 1000.)) for t in [0] + list(bins)]) / float(intervals.size)

    def autocorrelation(self, bin_size, bin_count):
        """
        :param bin_size: int, size of a bin in the histogram, in seconds
        :param bin_count: int, number of bins
        :return: x, y
        """
        # page 28 textbook
        vals = [0] * bin_count
        for t1 in self.spikes:
            for t2 in self.spikes:
                m = int(np.floor(np.abs(t1 - t2) / bin_size))
                if m < bin_count:
                    vals[m] += 1
        v = (np.asarray(vals, dtype=np.float) / self.duration)
        v -= (len(self.spikes) * len(self.spikes) * bin_size) / (self.duration * self.duration)
return np.arange(0, bin_count) * int(bin_size * 1000), v





#%%

"""
Important steps for make an attractor of real data 

1. Know the total acquisition time & the frames time 
2. Consider study cell for individuality 
3. Depending on the ODF equations that you will apply, modify the correspondant
variables. In this case, Røssler equations have three variables: a,b,c
4. Is important to know the steps separation between each change in the signals
In this case, between spikes.
5. Calculate the speed of change or rate of change in your time series 

"""
def rossler_attractor (x,y,z, a=0.1,b=0.1, c=0.14):
    x_dot =-y-z
    y_dot = x+a*y
    z_dot = b+z*(x-c)
    
    return x_dot,y_dot,z_dot


step_size = 0.1
steps =100000

xx=np.empty((steps + 1))
yy=np.empty((steps + 1))
zz=np.empty((steps + 1))

xx[0],yy[0],zz[0] = (0.1,0.,0.1)

for i in range(steps) : 
    x_dot, y_dot, z_dot = rossler_attractor(xx[i], yy[i], zz[i])
    
    xx[i + 1] = xx[i] + (x_dot * delta_t)
    yy[i + 1] = yy[i] + (y_dot * delta_t)
    zz[i + 1] = zz[i] + (z_dot * delta_t) 
    


#%%

# Try # 1000 of plot and calculate an atracttor 



# Funcion para atractor de lorenz

def Lor(X,t):
    x,y,z=X
    return np.array([s*(y-x),
            r*x-y-x*z,
            x*y-b*z])

#%%




def JacLor(t,X):
    x,y,z=X
    Jac=np.array([[-s, s, 0],
                  [r-z, -1, -x],
                  [y, x, -b]])
return Jac
#%%


def plotBranch(ax,Cells,farg='k-',width=2,maxgap=0.05,hilo=False):
    distances=np.sqrt(np.diff(Cells[:,300])**2 + np.diff(Cells[:,300])**2)
    limits=np.where(distances>maxgap)[0] +1 
    limits=np.append(limits,len(Cells))
    limits=np.insert(limits,0,0)
    for inic,fin in zip(limits[:-1],limits[1:]):
        ax.plot(Cells[inic:fin,3000],Cells[inic:fin,3000],farg,lw=width)
        if hilo:
            ax.plot(Cells[inic:fin,300],Cells[inic:fin,100],farg,lw=width)
            
            return limits
        
Branch = plotBranch(time,Cell1)





plt.figure(1,figsize=(13,4))
plt.clf()

#ax1=plt.subplot2grid((3,2),(0,0))
ax1=plt.subplot2grid((1,7),(0,0),colspan=2)
#ax1=plt.subplot(131)

ax1.plot(time[:],Cell1[0:3000],'k')
#ax1.plot((72,77),(Vthresh,Vthresh),'r-')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('F-F0 (a.u.)')

ax2=plt.subplot2grid((1,7),(0,2),colspan=2)

ax2.plot(ISI222[:,0]/1000,ISI222[:,1],'k.',ms=3)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('intervals (ms)')

ax3=plt.subplot2grid((1,7),(0,4),colspan=3,projection='3d')

ax3.plot(Vtraces[:,1],np.zeros_like(Vtraces[:,2]),Vtraces[:,3],lw=0.5,alpha=0.5)
ax3.plot(np.zeros_like(Vtraces[:,1]),Vtraces[:,2],Vtraces[:,3],lw=0.5,alpha=0.5)
ax3.plot(Vtraces[:,1],Vtraces[:,2],np.zeros_like(Vtraces[:,3]),lw=0.5,alpha=0.5)
ax3.plot(Vtraces[:,1],Vtraces[:,2],Vtraces[:,3],'k')

ax3.azim=50
ax3.zaxis.set_rotate_label(False) 
ax3.set_xlabel('$\mathsf{a_{sd}}$',fontsize='x-large')
ax3.set_ylabel('$\mathsf{a_{sr}}$',fontsize='x-large')
ax3.set_zlabel('$\mathsf{a_{h}}$',fontsize='x-large',rotation=90)
ax3.set_xticks((0,0.2,0.4,0.6))
ax3.set_yticks((0,0.2,0.4,0.6))
ax3.set_zticks((0,0.05,0.1,0.15))


plt.figtext(0.01,0.92,'A',fontsize='xx-large')
plt.figtext(0.29,0.92,'B',fontsize='xx-large')
plt.figtext(0.6,0.92,'C',fontsize='xx-large')

plt.tight_layout()

#%% 


""" 
Comparing one cell response with another 

"""
import fbprophet 

Cell1 = datosNormFilt[0,200,:].T 
   
Cell2 = datosNormFilt[0,56,:].T


Cells_comp = pd.DataFrame({'Cell1':Cell1, 'time':time})

Cells_comp2 = Cells_comp.rename(columns={'Cell1':'ds', 'time':'y'})

Cells_comp2['y'] = np.log(Cells_comp2['y'])

Cells_prophet = fbprophet.Prophet(changepoint_prior_scale=0.20)
Cells_prophet.fit(Cells_comp2)


#%%

"""
Analisis de distancia cortical de acuerdo a valores de intensidad de 
 imagenes T1, con denoise y BiasField. LOB izq, der"
"""

filename ="values27A21d"
Values =np.loadtxt(filename + ".csv",delimiter=',')

# Data partition 

f=len(Values)
ini = f//2

Val=np.array([Values[i:i+ini] for i in range(0,f,ini)])

# Left and right data selection i=0, left / i = 1 right

i=0

x = Val[i,:,0] # distance en mm
y = Val[i,:,1] #voxel value

dy = np.zeros(y.shape, np.float)
dy[0:-1] = np.diff(y)/np.diff(x)
dy[-1] = (y[-1]-y[-2])/(x[-1] - x[-2]).T


plt.figure(0)
plt.clf()

plt.subplot(211)
plt.plot(x,y)
plt.plot(x,y,'r.')

plt.subplot(212)
plt.plot(x,dy)
plt.plot(x,dy,'k.')

# When my data are flip, decomment line 1163

#dy = np.flip(dy,0)
#x = np.flip(x,0) # distance en mm

#%%

#Select the positive dy values

dDyPos = np.array(np.where(dy>0))
dDy = dy[dy>0]
dDyPos = dDyPos[0,:]

ddyNeg = np.array(np.where(dy<0))
ddy = dy[dy<0]
ddyNeg = ddyNeg[0,:]

# Interval value selection to take the cortical distanc

"""
INICIO
"""

FstP= np.amax(ddy)
i = np.array(np.where(dy == FstP))
i = i[0,0]

FromHere = dy[i:]  #dy array segmentation. FstPOINT

"""
FINAL
"""
LastP = np.amax(FromHere)
j = np.array(np.where(dy == LastP))
j = j[0,0]

# From x array, take the correspondant interval (mm)

InterVal= x[8:23] 
Distance = InterVal[-1] - InterVal[0]   

#_____________________________________________________________________________
#FstPos = np.array(np.where(dy > 0))
#i = FstPos[0,0] # First negative value position
#LstPos = np.array(np.where(dy == np.amax(dy)))
#def derivative(f,a,method='central',h=0.01):
#    if method == 'central':
#        return (f(a + h) - f(a - h))/(2*h)
#    elif method == 'forward':
#        return (f(a + h) - f(a))/h
#    elif method == 'backward':
#        return (f(a) - f(a - h))/h 
#    else:
#        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
#        
#
#derivative(Val[i],Val[i])
#%%

"""
GRAFICA DE GROSOR CORTICAL: derecha e izquierda
# in my folder data: /misc/carr/aaquiles/MRI
"""
df = pd.read_csv("GrososrCortical.csv") 

sns.set(font_scale = 1.3)

g =sns.factorplot(x = "Days", y= "Cortical Thickness Right (mm)", hue ="Group", 
               col = "Sex",  data = df, palette="Set1", legend =False)

g.despine(left=True)

g.set_xlabels("Post treatment days")
g.set_ylabels("Cortical Thickness Right (mm)")
plt.legend(loc='upper left')

#%%

"""
Transform matrix of anatomical images T1 
# in my folder data: /misc/carr/aaquiles/MRI/TranfMat/Control *or BCNU
"""
import os

ESTO = "T1_bet_to_std_12F121d_mat_hd.xfm"
base = os.path.splitext(ESTO)[0]
os.rename(ESTO, base + ".csv")

MyFile = "T1_bet_to_std_12F121d_mat_hd.csv"
Matriz = pd.read_csv(MyFile)

M1 = Matriz.rename(index = str, columns =
                   {"-center         0.00000    0.00000    0.00000":"Values"})

M2 = M1.Values.str.split(expand=True,)

Scale = np.array(M2.iloc[2,1:4]).astype(float)
MeanScale = np.mean(Scale) 

#%%

"""
NORMALIZATION of cortical thickness metrics based on the transformation scale 
MEAN values 
""" 

TransScales_R = df[['Cortical Thickness Right (mm)','Transformation Scale']]
TransScales_L = df[['Cortical Thickness Left (mm)','Transformation Scale']]

Extract = np.array(TransScales_R.iloc[:,0:32])

Rate_R = np.product(Extract,axis=1)

df.insert(7,"Scale_Rate", Rate_R,True)

g =sns.factorplot(x = "Days", y= "Scale_Rate", hue ="Group", 
               col = "Sex",  data = df, palette="Set1", legend =False)

g.despine(left=True)

g.set_xlabels("Post treatment days")
g.set_ylabels("Right Cortical Thickness_NORM")
plt.legend(loc='upper left')
























