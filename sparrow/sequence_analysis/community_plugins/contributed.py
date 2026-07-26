from sparrow.sequence_analysis.plugins import BasePlugin


class MultiplicativeFCR(BasePlugin):
    def __init__(self, protein):
        super().__init__(protein)

    def calculate(self, factor=2.0):
        """
        This analysis scales the FCR (fraction of charged residues) of the
        protein by a constant factor - doubling it by default. This is a simple
        example of a contributed plugin.

        Parameters
        ----------
        factor : float
            The factor by which the FCR will be multiplied (default is 2.0)

        Returns
        -------
        float
            The protein's FCR multiplied by ``factor``.
        """
        return factor * self.protein.FCR
