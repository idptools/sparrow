from sparrow.patterning import scd
import numpy as np

def compute_nu_zheng2020(seq):
    """
    Function takes in a sequence and returns a calculate Nu scaling value 
    from Sequence Hydropathy Decoration (SHD) and Sequence Charge Decoration)
    
    Nu = -0.0423×SHD + 0.0074×SCD+0.701
    
    This equation for predicting nu is adopeted from Zheng et al. [1].

    Parameters
    ------------------
    seq : str
        Amino acid sequence (must be valid amino acids only)

    Returns
    ------------------
    float
        Returns the predict scalinge exponent (nu), a dimensionless 
        parameter which should fall between 0.33 and 0.6 (in theory).

    References
    ---------------
    [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. 
    Hydropathy Patterning Complements Charge Patterning to Describe 
    Conformational Preferences of Disordered Proteins. J. Phys. 
    Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288

    """

    SHD = scd.compute_shd(seq)
    SCD = scd.compute_scd_x(seq)
        
    # calculate Nu from SHD and SCD
    nu = (-0.0423*SHD)+(0.0074*SCD)+0.701
    
    return nu



def compute_rg_zheng2020(seq):
    """
    Function that takes in an amino acid sequence and computes the 
    expected radius of gyration using the nu-dependent Rg as developed by 
    Zheng et al.

    Parameters
    ------------------
    seq : str
        Amino acid sequence (must be valid amino acids only)

    Returns
    ------------------
    float
        Returns the predict radius of gyration in Angstorms

    References
    ---------------
    [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. 
    Hydropathy Patterning Complements Charge Patterning to Describe 
    Conformational Preferences of Disordered Proteins. J. Phys. 
    Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288
    """
    nu = compute_nu_zheng2020(seq)
    
    gamma = 1.1615
    b = 5.5 # note in Angstroms instead of nanometers
    N = len(seq)

    numerator = gamma*(gamma+1)

    denominator = 2*(gamma+2*nu)*(gamma+2*nu+1)

    return np.sqrt(numerator/denominator)*b*np.power(N,nu)
    
