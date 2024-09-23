# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:08:59 2024

@author: Pepinaldo
"""

import numpy as np
import pandas as pd

data = pd.read_excel("E:/ciencias de datos/2024/alc/tp/matrizlatina2011_compressed_0.xlsx",sheet_name=1)

#####
Cri = data.drop([col for col in data.columns if not col.startswith("CRI") and not col.startswith("Country_iso3")], axis = 1)
#####
#COSTA RICA - COSTA RICA COMO DATAFRAME LIMPIO Y COMO ARRAY DE NUMPY
CriCri = Cri[(Cri["Country_iso3"]=="CRI")]
CriCri = CriCri.reset_index(drop=True).drop([col for col in CriCri.columns if col.startswith("Country")],axis = 1)
CriCri
npCri = CriCri.to_numpy()
npCri
#####
Nic = data.drop([col for col in data.columns if not col.startswith("NIC") and not col.startswith("Country_iso3")], axis = 1)
#####

#NICARAGUA - NICARAGUA COMO DATAFRAME LIMPIO Y COMO ARRAY DE NUMPY
NicNic = Nic[(Nic["Country_iso3"]=="NIC")]
NicNic = NicNic.reset_index(drop=True).drop([col for col in NicNic.columns if col.startswith("Country")],axis = 1)
NicNic
npNic = NicNic.to_numpy()
npNic

#####
#COSTA RICA - NICARAGUA COMO DATAFRAME LIMPIO Y COMO ARRAY DE NUMPY
CriNic = Nic[(Nic["Country_iso3"]=="CRI")]
CriNic = CriNic.reset_index(drop=True).drop([col for col in CriNic.columns if col.startswith("Country")],axis = 1)
CriNic
npCriNic = CriNic.to_numpy()
npCriNic
#####

#####
#NICARAGUA - COSTA RICA COMO DATAFRAME LIMPIO Y COMO ARRAY DE NUMPY
NicCri = Cri[(Cri["Country_iso3"]=="NIC")]
NicCri = NicCri.reset_index(drop=True).drop([col for col in NicCri.columns if col.startswith("Country")],axis = 1)
NicCri
npNicCri = NicCri.to_numpy()
npNicCri

#####

#P1 es Costa Rica y P2 es Nicaragua
total = data.drop([col for col in data.columns if not col.startswith("Output") and not col.startswith("Country_iso3")], axis = 1)
total

###
#TOTAL DE CAPITALES PARA COSTA RICA EN DATAFRAME Y EN ARRAY NUMPY
totalCRI = total[(total["Country_iso3"]=="CRI")].reset_index(drop=True)
totalCRI = totalCRI.drop([col for col in totalCRI.columns if col.startswith("Country")],axis = 1)
totalCRI
nptotalCRI = totalCRI.to_numpy()
nptotalCRI

###
#TOTAL DE CAPITALES PARA NICARAGUA EN DATAFRAME Y EN ARRAY NUMPY
totalNIC = total[(total["Country_iso3"]=="NIC")].reset_index(drop=True)
totalNIC = totalNIC.drop([col for col in totalNIC.columns if col.startswith("Country")],axis = 1)
totalNIC
####
nptotalNIC = totalNIC.to_numpy()
nptotalNIC
###

#para armar Z y luego A usamos lo siguiente:

arriba = np.hstack((npCri,npCriNic))
abajo = np.hstack((npNicCri,npNic))
ZMATRIX = np.vstack((arriba,abajo))
ZMATRIX
"""Esta es la matriz Z de flujo de capitales de 
manera intrarregional e interregional para los sectores 
P1 = Costa Rica y P2 = Nicaragua
"""
####
#la diagonal tiene los valores del total de capitales por cada fila de esta.
def Idxm(total):   
  n = total.shape[0]
  Id = np.eye(n)
  for i in range(len(Id)):
    for j in range(len(Id[i])):
      if i == j :
        Id[i][j] = Id[i][j] * total[j]
        if Id [i][j] == 0:
            Id[i][j] = 1

  return(Id)

#####
IdentidadCRI = Idxm(nptotalCRI)
IdentidadCRI
#####
IdentidadNIC = Idxm(nptotalNIC)
IdentidadNIC

##### 
ANic = npNic @ np.linalg.inv(IdentidadNIC)
ANic
#####
ACri = npCri @ np.linalg.inv(IdentidadCRI)
ACri
#####

ACriNic = npCriNic @ np.linalg.inv(IdentidadNIC)
ACriNic
#####
ANicCri = npNicCri @ np.linalg.inv(IdentidadCRI)
ANicCri 
#####

AC = np.hstack((ACri,ACriNic))
AN = np.hstack((ANic,ANicCri))
A = np.vstack((AC,AN))
A
"""
Y aquí tenemos como resultado a la Matriz A, con los coeficientes tecnicos
que representan las proporcionalidades de produccion que existen en las relaciones
de forma intra e interregional sobre P1 y P2
"""
#DE LA ECUACIÓN 4 DE LEONTIEF PARA RELACIONES DE DOS REGIONES
#CALCULO LA DEMANDA EXTERNA


dCri = (np.eye(40) - ACri) @ nptotalCRI
dCri

#un shock del 10% negativo sobre el sector 5 de este sería agarrar dCri y a su
#indice 4 que representa al sector 5 restarle dCri[4]*0.1 y al resto restarle 0
# y luego calcular Lontief con el nuevo dCri'
#analogamente con los sectores s06,s07 y s08 sumandole un 3.3% de lo q representan

#hecho de forma pedorra tal vez

""" ESTA FUNCION CALCULARA EL SHOCK PERO CON UN RAZONAMIENTO DE IR TRABAJANDO POR
PARTES, NO ES DEFINITIVO, VER BIEN DESPUES!"""

def diferencialShock(d):
    n = d.shape[0]
    m= d.shape[1]
    
    #construyo al diferencial
    dif = np.zeros((n,m))
    #B = np.zeros((n,m))
    
    dif[4] = d[4] * -0.1
    dif[5] = d[5] * 0.033
    dif[6] = d[6] * 0.033
    dif[7] = d[7] * 0.033
    
    #A resulta ser el diferencialD que crea a d' junto al d introducido.
    
    dPrima = d + A
    
    return dPrima
        
dPrimaCri = diferencialShock(dCri)
dPrimaCri


#utilizar las inversas mostradas en clase please!
#Uso lo que me brinda la ecuacion 5!!!
newP1CRI = np.linalg.inv((np.eye(40) - ACri))@ dPrimaCri
newP1CRI

#AHORA UTILIZAR LA ECUACION 6 DE LA RELACION MULTIREGIONAL 
#plantear si usar la matriz A tambien, hacerlo despues

"""LA IDEA ES CREAR UNA FUNCION IGUALMENTE QUE GENERALICE TODO LO ANTERIOR, 
POR FAVOR HACERLA MAÑANA LUNES!!!!!!!"""

