import numpy as np  
from utils import savefig, save_pickle
import sys  
sys.path.append('code/')
from ricatti import get_propagator, get_modes, plot_eigenvector, plot_solution
import matplotlib.pyplot as plt
    
eigs = dict()
N = 40
for shape in ['disk','triangle','disks']:   
    for bc in ['Dirichlet','Neumann']:  
        P, doubleCell, V0, V2 = get_propagator(N, shape ,bc)
    
        eigenvalues, eigenvectors = get_modes(P)
            
        for i in range(5):
            plot_eigenvector(doubleCell, P, V0, V2, eigenvectors, eigenvalues, i)
            savefig("FIGS/eigenmodes_"+shape+"_"+bc+"_"+str(i)+".png")
        plt.close('all')

        x0 = np.real(eigenvectors[:,0])
        x0 = np.ones_like(eigenvectors[:,0])
        plot_solution(doubleCell, P, V0, V2, x0)
        savefig("FIGS/difference_"+shape+"_"+bc+".png")
        plt.close('all')
            
        eigs[shape+"_"+bc] = eigenvalues[:6]
            
save_pickle(eigs,"FIGS/eigs.pkl")
