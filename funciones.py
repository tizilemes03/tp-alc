# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:29:14 2024

@author: nacha
"""

import numpy as np
import pandas as pd

def intercambiarfilas(A, fila1, fila2):
    A[[fila1, fila2]] = A[[fila2, fila1]]
    return A

def calcularLU(A):
    m, n = A.shape
    if m != n:
        print('Matriz no cuadrada')
        return
    
    P = np.arange(n)  # Inicializa el vector de permutaciones
    Ac = A.copy()
    
    # Recorremos la matriz para realizar la eliminación gaussiana
    for fila in range(m):
        if Ac[fila, fila] == 0:
            # Si el pivote es cero, intercambiamos con la siguiente fila
            if fila + 1 < m:  # Asegúrate de que no estamos en la última fila
                intercambiarfilas(Ac, fila, fila + 1)
                P[fila], P[fila + 1] = P[fila + 1], P[fila]
            else:
                print("La matriz no tiene factorización LU.")
        
        # Actualizamos la eliminación gaussiana
        for i in range(fila + 1, m):
            factor = Ac[i, fila] / Ac[fila, fila]
            Ac[i, fila] = factor  # Guardamos el factor de eliminación en L
            Ac[i, fila + 1:] -= factor * Ac[fila, fila + 1:]

    # Construimos las matrices L y U
    L = np.tril(Ac, -1) + np.eye(m)  # Parte triangular inferior más la identidad
    U = np.triu(Ac)  # Parte triangular superior

    return L, U, P

A = np.array([[0.7, 0, -0.1],
     [-0.05, 0, -0.2],
     [-0.1, -0.15, 0.9]])
L, U, P=calcularLU(A)

def inversaU(U):
    n = U.shape[0]
    U_inv = np.zeros_like(U, dtype=float)
    # Sustitución hacia atrás, recorre las filas de abajo hacia arriba
    for i in range(n-1, -1, -1):
        # Comenzamos con la diagonal, la diagonal de la inversa de una matriz triangular es 
        #simplemente el recíproco de la diagonal de la matriz original
        U_inv[i, i] = 1.0 / U[i, i] 
        #recorre las columnas a la derecha del elemento diagonal en la fila i
        for j in range(i+1, n): 
            U_inv[i, j] = -U[i, j] * U_inv[j, j] / U[i, i] 
            #elementos no diagonales
            for k in range(i+1, j):
                U_inv[i, j] -= U[i, k] * U_inv[k, j] / U[i, i]
    return U_inv

def inversaL(L):
    n = L.shape[0]
    L_inv = np.zeros_like(L, dtype=float)
    # Sustitución hacia adelante para calcular cada columna de L_inv
    for i in range(n):
        # Comenzamos con la diagonal
        L_inv[i, i] = 1.0 / L[i, i]
        for j in range(i):
            suma = 0
            for k in range(j, i):
                suma += L[i, k] * L_inv[k, j]
            L_inv[i, j] = -suma / L[i, i]
    
    return L_inv

def inversaLU(L, U, P):
    Inv = np.linalg.inv(U)@np.linalg.inv(L)
    Inv_permutado = np.zeros_like(Inv)
    for i in range(len(P)): 
        Inv_permutado[:, P[i]] = Inv[:, i] 
    return Inv_permutado
    

inversa = inversaLU(L, U, P)
d = np.array([[100], [100], [300]])
d2 = np.array([[100], [100], [301]])
d_permutado = d[P]
d_permutado2 = d2[P]
p = inversa@d_permutado
p2 = inversa@d_permutado2

C = np.array([[0.65, 0, 0],
              [-0.05, 0.5, -0.15],
              [-0.2, -0.3, 0.45]])
print(np.linalg.inv(C))

#FUNCIONES DEL PUNTO 7

data = pd.read_excel("E:/ciencias de datos/2024/alc/tp/matrizlatina2011_compressed_0.xlsx",sheet_name=1)

"""
Para poder crear la matriz de flujo de capitales intrarregionales e interregionales de Costa Rica y Nicaragua
seguimos utilizamos los siguientes métodos de las librerias propiciadas.
"""

def generadorMatrizZ(data,PAIS1,PAIS2):
    #PAIS 1 corresponde a filas y PAIS 2 corresponde a columnas
    Columnas = data.drop([col for col in data.columns if not col.startswith(PAIS2) and not col.startswith("Country_iso3")], axis = 1)
    FilasYColumnas = Columnas[(Columnas["Country_iso3"]== PAIS1 )]
    Matriz = FilasYColumnas.reset_index(drop=True).drop([col for col in FilasYColumnas.columns if col.startswith("Country_iso3")],axis = 1)
    
    return Matriz

CriCri = generadorMatrizZ(data,"CRI","NIC")

NicNic = generadorMatrizZ(data,"NIC","NIC")

CriNic = generadorMatrizZ(data,"CRI","NIC")

NicCri = generadorMatrizZ(data,"NIC","CRI")

def produccionesPais(data,PAIS):
    total = data.drop([col for col in data.columns if not col.startswith("Output") and not col.startswith("Country_iso3")], axis = 1)
    totalPAIS = total[(total["Country_iso3"]==PAIS)].reset_index(drop=True)
    totalPAIS = totalPAIS.drop([col for col in totalPAIS.columns if col.startswith("Country")],axis = 1)
    totalPAIS = totalPAIS.to_numpy()
    return totalPAIS

pCRI = produccionesPais(data,"CRI")

pNIC = produccionesPais(data,"NIC")

#para armar Z y luego A usamos lo siguiente:

def ZGrande(ZP1P1,ZP1P2,ZP2P1,ZP2P2):
    arriba = np.hstack((ZP1P1,ZP2P2))
    abajo = np.hstack((ZP2P1,ZP2P2))
    ZMatriz = np.vstack((arriba,abajo))
    return ZMatriz

"""Esta es la matriz Z de flujo de capitales de 
manera intrarregional e interregional para los sectores 
P1 = Costa Rica y P2 = Nicaragua"""
Z = ZGrande(CriCri,CriNic,NicCri,NicNic)

####
#la diagonal tiene los valores del total de capitales por cada fila de esta.
def IdxP(pPAIS):   
  n = pPAIS.shape[0]
  Id = np.eye(n)
  for i in range(len(Id)):
    for j in range(len(Id[i])):
      if i == j :
        Id[i][j] = Id[i][j] * pPAIS[j]
        if Id [i][j] == 0:
            Id[i][j] = 1

  return(Id)

#####
IdPCRI = IdxP(pCRI)
#####
IdPNIC = IdxP(pNIC)

##### 
def AInsumoProductoMultiRegional(ZP1P1,ZP1P2,ZP2P1,ZP2P2,IdP1,IdP2):
    
    AP1P1 = ZP1P1 @ np.linalg.inv(IdP1)
    
    AP1P2 = ZP1P2 @ np.linalg.inv(IdP2)
    
    AP2P1 = ZP2P1 @ np.linalg.inv(IdP1)
    
    AP2P2 = ZP2P2 @ np.linalg.inv(IdP2)
    
    AUp = np.hstack((AP1P1,AP1P2))
    
    Adown = np.hstack((AP2P1,AP2P2))
    
    A = np.vstack((AUp,Adown))
    
    return A
    
    
A = AInsumoProductoMultiRegional(CriCri,CriNic,NicCri,NicNic,IdPCRI,IdPNIC)

#AHORA PARA EL PUNTO DEL SHOCK
#la demanda trabaja desde la fila 1 a la 40 sobre la region de Costa rica, y de la 41 a la 80 sobre Nicaragua
#EL SECTOR 
def demandaDeA(A,p1,p2):
    idA = np.eye(A.shape[0])
    pTotal= np.vstack((p1,p2)) 
    d = (idA - A) @ pTotal
    return d 

dA = demandaDeA(A,nptotalCRI,nptotalNIC)
dA

pTotal= np.vstack((nptotalCRI,nptotalNIC)) 

def diferencialShock(A,d):
    n = d.shape[0]
    m= d.shape[1]
    idA = np.eye(A.shape[0])
    #construyo al diferencial
    dif = np.zeros((n,m))
    #B = np.zeros((n,m))
    
    dif[4] = d[4] * -0.1
    dif[5] = d[5] * 0.033
    dif[6] = d[6] * 0.033
    dif[7] = d[7] * 0.033
    
    #A resulta ser el diferencialD que crea a d' junto al d introducido.
    
    dPrima = d + dif
    #veo si lo pongo o no
    pTotalPrima = np.linalg.inv(idA - A)@dPrima 
    
    return pTotalPrima

nuevoPTotal = diferencialShock(A,dA)


