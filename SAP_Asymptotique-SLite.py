#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# # Calcul asymptotique d'une CRPA N éléments SAP

# ## Définition du scénario

# Nombre d'antennes
N = st.sidebar.slider('Nombre Antennes',min_value=2,max_value=64,step=1)
st.write("Nombre d'Antennes = ",N)

st.latex(r''' s_i = e^{2pi \frac{d} {\lambda} sin(\theta)} ''')
st.latex(r''' u_i = \frac{d} {\lambda} sin(\theta) ''') 

# Direction de brouillage
ui = st.sidebar.slider('Direction du Brouilleur',min_value=-1.0,max_value=1.0,step=0.01)
st.write('Direction de brouillage = ',ui)

# Directions de brouillage en angle (P.24 Guercy)
# Angles en degrés avec 0° au zénith

# Puissance de brouillage (en dB)
INR = st.sidebar.slider('Puissance du Brouilleur',min_value=0.0,max_value=100.0,step=10.0)
st.write('Puissance de brouillage',INR)

INR_lin = pow(10,(INR/10))

# Définition du réseau linéaire, espacement lambda/2
n = np.arange(-N/2, N/2)
n = n.reshape(N,1)
n = n + 0.5

# Vecteurs de pointages vers les Brouilleurs J
Vj = np.exp(1j*n*np.pi*ui)

# Contrainte Spatiale ADirectionnelle
Cs = np.zeros((N,1))
Cs[1]=1

# Calcul des pondérations
# ## Calcul de la matrice d'intercorrélation idéale
st.latex(r''' S_n = INR_{LIN} * (V_j*V_j^{-H}) + I_n * sigma^2 ''') 

# Calcul de la matrice d'intercorrélation idéale
Sn = INR_lin*np.dot(Vj,np.conj(Vj).T) + np.identity(N)

# Calcul des pondérations
st.latex(r''' w = \frac {S_n^{-1} * C_s} {C_s^{H} * S_n^{-1} * C_s} ''')
# 
# ou
# 
st.latex(r''' w = \frac {z_n} {C_s^{H} * z_n}''')
st.write('en posant')
st.latex(r'''  z_n = S_n^{-1} * C_s ''')

# Calcul des pondérations
zn = np.linalg.solve(Sn, Cs)
w = zn / np.dot(np.conj(Cs).T, zn)

# TO DO
# Affichage des pondérations
# st.write(w)


# Evaluation des performances
st.header('Evaluation des performances')
st.latex(''' R_{ej} = w^H * V_j  ''')
st.write('avec') 
st.latex(''' V_j ''')
st.write('Vecteur de pointage vers Brouilleur ''')

# Evaluation des performances
Rej = 20*np.log10(abs(np.dot(np.conj(w).T,Vj)))

# Toutes les réjections

st.write('Réjection pour chacun des brouilleurs : ')
Rej

# Réjection du 1er brouilleur
# TODO : Amélioration l'affichage sytle :.0f
st.write('Réjection du 1er brouilleur = ', Rej[0][0],'dB')

# ## Diagramme d'antenne résultant
st.title('Diagramme')
# Obtenu en calculant pour toutes les directions de l'espace
st.latex(''' W^H * U_{dir} ''')
st.write('avec') 
st.latex('''  U_{dir} ''')
st.write('Vecteur de pointage vers chacune des directions spatiales')

# Vecteur de pointage dans toutes les directions à évaluer
Udir = np.arange(-1, 1,0.01)
Udir = Udir.reshape(200,1)

# Calcul de la réjection dans toutes les directions
Rejection = np.zeros((200,1))
ii = 0
for element in Udir:
    U = np.exp(1j*n*np.pi*element)
    Rejection[ii] = 20*np.log10(abs(np.dot(np.conj(w).T,U)))
    ii = ii+1

# ## SINR
st.title('SINR')
# ## Calcul du SINR *asymptotique* en LINEAIRE
st.header('SINR Asymptotique')
st.latex(''' SINR = 10*log_{10}(C_s^H . S_n^{-H} . C_s) ''')

# ## Calcul du SINR *asymptotique* en dB
10*np.log10(abs(np.dot(np.conj(Cs).T, np.conj(zn))))[0][0]


# ## Calcul du SNR mono-antenne
st.header('SNR Mono-Antenne')
st.latex(r''' SNR_{MonoAntenne} = 10*log_{10} \frac{(norm(w^H*C_s)^2)} {w^H*S_n*w} ''')

#st.write('SNR_Mono = ',10*np.log10(abs(np.dot(np.conj(w).T,Cs))))
st.write('SNR par temps clair pour chacun des brouilleurs : ')
10*np.log10(abs(np.dot(np.conj(w).T,Cs)))

## TODO A mettre au propre
# SINR avec traitement
SINR = 10*np.log10(abs(np.dot(np.conj(Cs).T, np.conj(zn))))[0][0]
#print(f'SINR = {SINR:.2f}dB')
st.write('SINR = ', SINR,'dB')

# SNR en absence de brouillage et avec bruit thermique uniquement
SNRopt = 10*np.log10(np.dot(np.conj(Cs).T,Cs))[0][0]
st.write('SNRopt =', SNRopt, 'dB')

# Dégradation due au traitement
SINR_Loss = SNRopt - SINR
#print(f'SINR_Loss = {SINR_Loss:.2f}dB')
st.write('SINR_Loss = ', SINR_Loss,'dB')

## FIGURES
fig = plt.figure()
axe = fig.add_axes([0,0,1,1])
axe.plot(Udir,Rejection)

axe.set_xlim([-1, 1])
#axe.set_ylim([-60, 20])
axe.title.set_text(f'Diagramme - Réseau {N} antennes - INR = {INR}dB - Brouillage {ui} - Loss = {SINR_Loss}')

# st.pyplot(fig)
st.plotly_chart(fig)
