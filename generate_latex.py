import numpy as np
from utils import load_pickle
import pyperclip
    
from decimal import Decimal, Context

def format_significant_digits(number, sig_digits):
    # Convertir le nombre en Decimal
    d = Decimal(str(number))
    
    # Créer un contexte avec la précision voulue
    context = Context(prec=sig_digits)
    
    # Appliquer le contexte et obtenir le nombre formaté
    formatted_number = context.create_decimal(d)
    
    # Retourner sous forme de chaîne, arrondi à 3 chiffres significatifs
    return str(formatted_number)

    
eigs = load_pickle("FIGS/eigs.pkl")

latex = ""
for shape in ['disk','triangle','disks']:   
    for bc in ['Dirichlet','Neumann']:  
        latex += r"""    
        \begin{figure}  
        \centering  
        """
        for i in range(5):  
            latex += r"""   
            \begin{subfigure}{0.3\linewidth}    
            \includegraphics[width=\linewidth]{FIGS/eigenmodes_"""+shape+"_"+bc+"_"+str(i)+".png}"
            lamb = format_significant_digits(np.real(np.log(eigs[shape+"_"+bc][i])),3)
            latex += r"""   
            \caption{$\lambda_{"""+str(i)+r"""}="""+lamb+"}\n"
            latex +=r"""\end{subfigure}\quad          """
        latex += r"""   
        \caption{Modes for a """+shape+""" with """+bc+""" boundary conditions on   
        the top and bottom wall.}
        \label{fig:"""+shape+"_"+bc+r"""}
        \end{figure}"""
print(latex)
            
            
pyperclip.copy(latex)
