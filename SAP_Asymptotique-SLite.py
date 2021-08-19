#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st

import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt

# # Calcul asymptotique d'une CRPA N éléments SAP

# ## Définition du scénario

# Nombre d'antennes
N = 16


# $ s_i = e^{2pi \frac{d} {\lambda} sin(\theta)}$ 
# 
# $ u_i = \frac{d} {\lambda} sin(\theta) $ 

# In[4]:


# Direction de brouillage
# ui = (-0.7,0.4,0.6)
# ui = (0.6)


# In[5]:


# Directions de brouillage en angle (P.24 Guercy)
# Angles en degrés avec 0° au zénith
ui = st.slider()'Direction d''arrivée',min_value=0.0,max_value=1.0,step=0.01)
st.write(ui)

# Puissance de brouillage (en dB)
INR = 40
INR_lin = pow(10,(INR/10))


# In[7]:


# Définition du réseau linéaire, espacement lambda/2
n = np.arange(-N/2, N/2)
n = n.reshape(N,1)


# In[8]:


# Vecteurs de pointages vers les Brouilleurs J
Vj = np.exp(1j*n*np.pi*ui)


# In[9]:


# Contrainte Spatiale ADirectionnelle
Cs = np.zeros((N,1))
Cs[1]=1


# # Calcul des pondérations
# 

# ## Calcul de la matrice d'intercorrélation idéale
# $ S_n = INR_{LIN} * (V_j*V_j^{-H}) + I_n $ 

# In[10]:


# Calcul de la matrice d'intercorrélation idéale
Sn = INR_lin*np.dot(Vj,np.conj(Vj).T) + np.identity(N)


# ## Calcul des pondérations
# $ w = \frac {S_n^{-1} * C_s} {C_s^{H} * S_n^{-1} * C_s} $
# 
# ou
# 
# $ w = \frac {z_n} {C_s^{H} * z_n} $ en posant $ z_n = S_n^{-1} * C_s $

# In[11]:


# Calcul des pondérations
zn = np.linalg.solve(Sn, Cs)
w = zn / np.dot(np.conj(Cs).T, zn)


# In[12]:


w


# ## Evaluation des performances

# $ R_{ej} = w^H * V_j $ avec $ V_j $ Vecteur de pointage vers Brouilleur

# In[13]:


# Evaluation des performances
Rej = 20*np.log10(abs(np.dot(np.conj(w).T,Vj)))


# In[14]:


Rej


# In[15]:


# Réjection du 1er brouilleur
print(f'Réjection = {Rej[0][0]:.0f}dB')


# ## Diagramme d'antenne résultant
# Obtenu en calculant pour toutes les directions de l'espace
# $ W^H * U_{dir} $ avec $ U_{dir} $ Vecteur de pointage vers chacune des directions spatiales

# In[16]:


# Vecteur de pointage dans toutes les directions à évaluer
Udir = np.arange(-1, 1,0.01)
Udir = Udir.reshape(200,1)


# In[17]:


Rejection = np.zeros((200,1))
ii = 0
for element in Udir:
    U = np.exp(1j*n*np.pi*element)
    Rejection[ii] = 20*np.log10(abs(np.dot(np.conj(w).T,U)))
    ii = ii+1


# In[18]:


fig = plt.figure()
axe = fig.add_axes([0,0,1,1])
axe.plot(Udir,Rejection)

axe.set_xlim([-1, 1])
#axe.set_ylim([-60, 20])
axe.title.set_text(f'Diagramme - Réseau {N} antennes - INR = {INR}dB - Brouillage {ui}')

# Marquage des directions de brouilleurs
ii = 0
for J in ui:
    plt.plot([J,J],[0,Rej[0][ii]], color ='red', linewidth=1.5, linestyle="--")
# Marquage de la réjection obtenue
    plt.annotate(f'{Rej[0][ii]:.0f}dB',
             xy=(J, Rej[0][ii]), xycoords='data',
             xytext=(+20, +40+ii*20), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ii = ii+1


# ## SINR

# ## Calcul du SINR *asymptotique* en LINEAIRE
# $ C_s^H * S_n^{-H} * C_s $

# In[19]:


# Calcul du SINR asymptotique LINEAIRE
# Cs.H*Sn-H*Cs 
abs(np.dot(np.conj(Cs).T, np.conj(zn)))[0][0]


# In[20]:


# Confirmation en passant par l'inverse de la matrice
abs(np.dot(np.conj(Cs).T, np.dot(np.conj(inv(Sn)),Cs)))[0][0]


# ## Calcul du SINR *asymptotique* en dB
# $ 10*log_{10}(C_s^H * S_n^{-H} * C_s) $

# In[21]:


10*np.log10(abs(np.dot(np.conj(Cs).T, np.conj(zn))))[0][0]


# ## Calcul du SNR mono-antenne
# $ 10*log_{10} \frac{(norm(w^H*C_s)^2)} {w^H*S_n*w} $

# In[22]:


10*np.log10(abs(np.dot(np.conj(w).T,Cs)))


# In[23]:


np.conj(w).T


# In[24]:


Cs


# In[ ]:




