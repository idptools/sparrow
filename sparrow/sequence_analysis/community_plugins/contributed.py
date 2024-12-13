from sparrow.sequence_analysis.plugins import BasePlugin


class DoubleFCR(BasePlugin):
    def __init__(self, protein):
        super().__init__(protein)

    def calculate(self, seq):
        """
        This analysis doubles the FCR (fraction of charged residues) of the protein.
        This is a simple example of a contributed plugin.

        Parameters
        --------------
        seq : sparrow.Protein
            A sparrow.Protein object instance

        Returns
        -------------
        float
            Returns the result of the contributed analysis

        """
        return 2.0 * self.protein.FCR
