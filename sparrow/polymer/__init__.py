## TO DO, comment! this

from . import scaling_parameters


class Polymeric:

    def __init__(self, protein_obj):        
        self.__protein = protein_obj

    
    def nu(self, mode='zheng2020'):
        return scaling_parameters.compute_nu_zheng2020(self.__protein.sequence)

    
    def radius_of_gyration(self, mode='zheng2020'):
        return scaling_parameters.compute_rg_zheng2020(self.__protein.sequence)


    
