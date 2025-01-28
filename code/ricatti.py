import numpy as np
from pymedit import P1Function, mmg2d, trunc
import scipy.sparse as sp
from pyfreefem import FreeFemRunner 
from utils import parseSysArgs, get_root, argDocString
import matplotlib.pyplot as plt     
    
from lib import generateStrip, verticesOnBoundary, set_debug
    
plt.ion()
    
# Verbosity tuning
debug = 5   
args = parseSysArgs()
ff_verbosity = int(args['-v'][0]) if '-v' in args else 0
set_debug(debug)
    
def get_propagator(N, shape, bc="Dirichlet"):
    """ bc: Neumann ou Dirichlet sur les parois de la bande """

    doubleCell, _ = generateStrip(2,N, shape=shape)

    # Rhs2: right boundary
    # Rhs4: left boundary
    def getBoundaryOperator(bdNum):
        bd2 = verticesOnBoundary(doubleCell, bdNum) 
        rhs2 = sp.csr_matrix(([1]*len(bd2),(bd2,range(len(bd2)))),shape=(doubleCell.nv,len(bd2)))
            
        script = "laplace.edp"
        runner = FreeFemRunner(get_root()+"/"+script,{'bc':bc},     
                               run_dir="run",debug=debug)
        runner.import_variables(Th=doubleCell,rhs2=rhs2)
        #A = runner.execute()['A']
        V2 = runner.execute(with_mpi=True,verbosity=ff_verbosity)['SOL']
        return V2
        
    V2 = getBoundaryOperator(2)
    V0 = getBoundaryOperator(4)
    
    # Restriction on middle boundary    
    V2r = V2[verticesOnBoundary(doubleCell, 5),:].todense()
    V0r = V0[verticesOnBoundary(doubleCell, 5),:].todense()

    # Fixed point scheme    
    P = np.zeros_like(V2r)
        
    residual = 1
    i = 0 
    while residual > 1e-20:
        Pnew = V0r+V2r@P@P    
        residual = np.linalg.norm(Pnew-P,np.inf)
        print("Residual = "+str(residual))
        P = Pnew
        i += 1
            
    print("Fixed point iteration converged in "+str(i)+" iterations !")
        
    return P, doubleCell, V0, V2
    
def get_modes(P):
    # Diagonalize P and sort eigenvalues according to the real part
    eigenvalues, eigenvectors = np.linalg.eig(P)
    order = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]    
    eigenvectors = eigenvectors[:,order]
    return eigenvalues, eigenvectors
    
def plot_eigenvector(doubleCell, P, V0, V2, eigenvectors, eigenvalues, i):
    mode = np.array(V2 @ P @ P @ eigenvectors[:,i] + V0 @ eigenvectors[:,i]).flatten()
    lamb = np.log(eigenvalues[i])
    exp = P1Function(doubleCell, lambda x : np.exp(lamb*x[0]))
    mode /= exp.sol
    P1Function(doubleCell, mode).plot()


def plot_eigenvectors(doubleCell, P, V0, V2, eigenvectors, eigenvalues):
    fig, ax = plt.subplots(2,3, figsize=(16,8)) 
    for i in range(6):    
        row, col = divmod(i, 3) 
        mode = np.array(V2 @ P @ P @ eigenvectors[:,i] + V0 @ eigenvectors[:,i]).flatten()
        lamb = np.log(eigenvalues[i])
        exp = P1Function(doubleCell, lambda x : np.exp(lamb*x[0]))
        mode /= exp.sol
        P1Function(doubleCell, mode) \
            .plot(fig=fig, ax=ax[row,col],title="Mode "+str(i) + " lambda="+format(np.log(eigenvalues[i]),".2f"))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.6)

def plot_solution(doubleCell, P, V0, V2, x0):
    # Solution to the infinite strip problem with u0 = 1
    fig, ax = plt.subplots(2,2, figsize=(16,8)) 
    sol1 = np.asarray(V0 @ x0 + V2 @ P @ P @ x0 )
    P1Function(doubleCell, sol1).plot(fig= fig, ax=ax[0,0], title="Solution for u=1 on left boundary")
    sol2 = np.asarray(V0 @ x0 )
    P1Function(doubleCell, sol2).plot(fig= fig, ax=ax[0,1], title="Solution for u=1 on left boundary and u=0 on right")
    diff = sol1 - sol2
    P1Function(doubleCell, diff).plot(fig= fig, ax=ax[1,0], title="Difference")

if __name__=="__main__":
    parser = argDocString("Compute the propagator operator in a neumann perforated "   
                          "strip for the Laplace and Stokes problems")
    parser.addArg("-shape","Shape (available: disk, triangle, duct, ellipse, "  
                           "square, duct, crescents)", meta="SHAPE",    
                  default="disk")
    parser.addArg("-N","Resolution","N", default=40)
    parser.addArg("-neumann","Use Neumann boundary conditions on the strip instead of Dirichlet")
    args = parser.parseSysArgs()

    N = args['-N']
    if "-neumann" in args:  
        bc = "Neumann"  
    else:   
        bc = "Dirichlet"
    P, doubleCell, V0, V2 = get_propagator(N, args['-shape'],bc)
        
    eigenvalues, eigenvectors = get_modes(P)
        
    plot_eigenvectors(doubleCell, P, V0, V2, eigenvectors, eigenvalues)

    x0 = np.real(eigenvectors[:,0])
    x0 = np.ones_like(eigenvectors[:,0])
    plot_solution(doubleCell, P, V0, V2, x0)
    import ipdb 
    ipdb.set_trace()

    input("Press any key")
