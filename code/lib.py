import numpy as np
from pymedit import P1Function, mmg2d, trunc
from pyfreefem import FreeFemRunner 
from utils import parseSysArgs
import scipy.signal as sg

sawtooth = lambda x :  sg.sawtooth(2*np.pi*(x))/2+0.5
debug = 5
args = parseSysArgs()
ff_verbosity = int(args['-v'][0]) if '-v' in args else -1
ncpu = int(args['-np'][0]) if '-np' in args else 1
wg = '-wg' in args
    
def set_debug(d):   
    global debug    
    debug = d
    

def hyperplane(A,B):    
    xA = A[0]   
    yA = A[1]   
    xB = B[0]   
    yB = B[1]   
        
    a = yA-yB  
    b = xB-xA  
    return lambda x : (x[0]-xA)*a+(x[1]-yA)*b

def generateHole(N, shape="disk"):
    code = """  
    IMPORT "io.edp" 
        
    mesh Th=square($N,$N, [x,y], flags=1);  
    exportMesh(Th);     
    """
    Th = FreeFemRunner(code).execute(config={'N':N},debug=debug)['Th']
    Th.debug = debug
        
    if shape=="disk":
        phi = lambda x : np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)-0.1
    elif shape=="triangle": 
        A = [0.3,0.2]  
        B = [0.7, 0.4]
        C = [0.4,0.7]
        phi = lambda x : max(hyperplane(B,A)(x),
                             hyperplane(C,B)(x),
                             hyperplane(A,C)(x))
    elif shape=="ellipse":  
        def phi(x): 
            theta= np.pi/3
            xR = np.cos(theta)*(x[0]-0.5)-np.sin(theta)*(x[1]-0.5)
            yR = np.sin(theta)*(x[0]-0.5)+np.cos(theta)*(x[1]-0.5)
            return xR**2/0.3**2+yR**2/0.1**2-1
    elif shape=="square":   
        phi = lambda x : max(np.abs(x[0]-0.5)-0.2,np.abs(x[1]-0.5)-0.2)
    elif shape=="duct": 
        bump = lambda x : np.cosh(3.2*x)/np.cosh(np.pi/2*np.sinh(3.2*x))**2 
        phi = lambda x : -max(x[1]-(0.81-0.3*bump(3*(0.7-sawtooth(x[0])))) 
                             ,-x[1]+(0.21+0.3*bump(3*(sawtooth(x[0])-0.3))))
    elif shape=="disks":    
        phi = lambda x : min(np.sqrt((x[0]-0.3)**2+(x[1]-0.5)**2)-0.1,    
                             np.sqrt((x[0]-0.7)**2+(x[1]-0.3)**2)-0.1,    
                             np.sqrt((x[0]-0.7)**2+(x[1]-0.7)**2)-0.1)
    elif shape=="crescents":    
        A = [0.2,0.5]   
        C = [0.8,0.8]   
        B = [0.8,0.2]
        p = 2
        phi = lambda x : max(0.2-x[0],hyperplane(C,B)(x),  
                             A[1]-x[1]+(x[0]-A[0])**p/(B[0]-A[0])**p*(B[1]-A[1]),
                             -A[1]+x[1]-(x[0]-A[0])**p/(C[0]-A[0])**p*(C[1]-A[1]))

    else:   
        raise Exception("Error: shape unknown (disk|triangle)")
    phiP1 = P1Function(Th,phi)
        
    hmin = 0.9/N
    hmax = 1.1/N
    hgrad = 1.3 
    hausd = 0.1*hmin
    paramsMMg = f""" 
    Parameters
    1
     
    10 Edges {hmin} {hmax} {hausd}
    """
        
    M = mmg2d(Th, hmin, hmax, hgrad, hausd, params=paramsMMg, nr=False, sol=phiP1,  
                          ls=True, extra_args="-nosurf")
    M.requiredEdges = np.where(np.isin(M.edges[:, -1], [1, 2, 3, 4]))[0]+1
    M.nre = len(M.requiredEdges)

    M = mmg2d(M, hmin, hmax, hgrad, hausd, params=paramsMMg, nr=False)

    if shape!="duct":
        M = trunc(M, 2)
    return M

def generateStrip(K, N, shape="disk"):
    Th = generateHole(N, shape)
    codeFreeFem=""" 
    IMPORT "io.edp"     
        
    mesh Th = importMesh("Th");     
    mesh Th2;   
    mesh Th3=Th; 
    for(int i = 1; i<$K; i++){
    Th2 = movemesh(Th,[x+i,y]);    
    Th3 = Th3+Th2;  
    }

    exportMesh(Th3); 
    """
    runner = FreeFemRunner(codeFreeFem,config={'K':K},debug=1)   
    runner.import_variables(Th=Th)  
    Th3=runner.execute(verbosity=ff_verbosity)['Th3']
        
    #Now remove internal spurious edges numbering 
    edgesTh3 = np.where(Th3.edges[:,-1]==2)[0]
    adjacentTris = [Th3.elemToTri(e) for e in Th3.edges[edgesTh3][:,:-1]]
    internalEdges = np.where(np.asarray([len(e) for e in adjacentTris])==2)[0]

    Th3.edges[edgesTh3[internalEdges],-1]=5   
    Th3._AbstractMesh__updateBoundaries()
    #Th3.N = K

    return Th3, Th
    

def verticesOnBoundary(Th, num):
    bd2 = np.unique(Th.verticesToEdges[:,Th.Boundaries[num]].tocoo().row)
    order = np.argsort(Th.vertices[bd2,1]) #Order vertices by y coordinate  
    return bd2[order]
