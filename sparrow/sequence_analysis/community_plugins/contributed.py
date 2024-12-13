from sparrow.sequence_analysis.plugins import BasePlugin


class MultiplicativeFCR(BasePlugin):
    def __init__(self, protein):
        super().__init__(protein)

    def calculate(self, factor=2.0):
        """
        This analysis doubles the FCR (fraction of charged residues) of the protein.
        This is a simple example of a contributed plugin.

        Parameters: factor (float)
        -------------
        factor: float
            The factor by which the FCR will be multiplied (default is 2.0)

        Returns
        -------------
        float
            Returns the result of the contributed analysis
        """
        return factor * self.protein.FCR
