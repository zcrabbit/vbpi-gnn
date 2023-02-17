
import numpy as np
import pdb

def decompJC(symm=False):
    # pA = pG = pC = pT = .25
    pden = np.array([.25, .25, .25, .25])
    rate_matrix_JC = 1.0/3 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -1.0
    
    if not symm:
        D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
        U_JC_inv = np.linalg.inv(U_JC)
    else:
        D_JC, W_JC = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0/pden))))
        U_JC = np.dot(np.diag(np.sqrt(1.0/pden)), W_JC)
        U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))
    
    return D_JC, U_JC, U_JC_inv, rate_matrix_JC


def decompHKY(pden, kappa, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa*(pA*pG+pC*pT))
    rate_matrix_HKY = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_HKY[i,j] = pden[j]
            if i+j == 1 or i+j == 5:
                rate_matrix_HKY[i,j] *= kappa
    
    for i in range(4):
        rate_matrix_HKY[i,i] = - sum(rate_matrix_HKY[i,])
    
    rate_matrix_HKY = beta * rate_matrix_HKY
    
    if not symm:
        D_HKY, U_HKY = np.linalg.eig(rate_matrix_HKY)
        U_HKY_inv = np.linalg.inv(U_HKY)
    else:
        D_HKY, W_HKY = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_HKY), np.diag(np.sqrt(1.0/pden))))
        U_HKY = np.dot(np.diag(np.sqrt(1.0/pden)), W_HKY)
        U_HKY_inv = np.dot(W_HKY.T, np.diag(np.sqrt(pden)))
       
    return D_HKY, U_HKY, U_HKY_inv, rate_matrix_HKY


def decompGTR(pden, AG, AC, AT, GC, GT, CT, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT))
    rate_matrix_GTR = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_GTR[i,j] = pden[j]
                if i+j == 1:
                    rate_matrix_GTR[i,j] *= AG
                if i+j == 2:
                    rate_matrix_GTR[i,j] *= AC
                if i+j == 3 and abs(i-j) > 1:
                    rate_matrix_GTR[i,j] *= AT
                if i+j == 3 and abs(i-j) == 1:
                    rate_matrix_GTR[i,j] *= GC
                if i+j == 4:
                    rate_matrix_GTR[i,j] *= GT
                if i+j == 5:
                    rate_matrix_GTR[i,j] *= CT
    
    for i in range(4):
        rate_matrix_GTR[i,i] = - sum(rate_matrix_GTR[i,])
    
    rate_matrix_GTR = beta * rate_matrix_GTR
    
    if not symm:
        D_GTR, U_GTR = np.linalg.eig(rate_matrix_GTR)
        U_GTR_inv = np.linalg.inv(U_GTR)
    else:
        D_GTR, W_GTR = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_GTR), np.diag(np.sqrt(1.0/pden))))
        U_GTR = np.dot(np.diag(np.sqrt(1.0/pden)), W_GTR)
        U_GTR_inv = np.dot(W_GTR.T, np.diag(np.sqrt(pden)))        
    
    return D_GTR, U_GTR, U_GTR_inv, rate_matrix_GTR