"""Core Protein module.

This module exposes the :class:`Protein` class, a lightweight container providing
on-demand computation of sequence-derived biophysical parameters and access to
predictors, polymeric properties, plugins, and sequence analysis utilities.

"""

from protfasta import utilities as protfasta_utilities

from sparrow import calculate_parameters, data, sparrow_exceptions
from sparrow.data import amino_acids
from sparrow.patterning import iwd, kappa, scd
from sparrow.sequence_analysis import (
    elm,
    patching,
    phospho_isoforms,
    physical_properties,
    sequence_complexity,
)
from sparrow.tools import general_tools, track_tools, utilities
from sparrow.visualize import sequence_visuals

__all__ = ["Protein"]


class Protein:
    def __init__(self, s, validate=False):
        """
        Construct for Protein object. Requires only a single sequence as
        input. Note that construction does not trigger any sequence
        parameters to be calculated, all of which are caculated as needed.

        See Also
        --------
        :class:`sparrow.sequence_analysis.plugins.PluginManager` : Plugin interface
        :class:`sparrow.predictors.Predictor` : Sequence-based predictors
        :class:`sparrow.polymer.Polymeric` : Polymer property calculations

        Parameters
        -------------
        s :  str
            Amino acid sequence

        validate : bool
            Flag that can be set if sequence should be validated to ensure
            it's a good amino acid sequence. Generally not necessary to
            sometimes useful. If set, the function constructor will
            automatically convert non-standard amino acids to standard

            amino acids according to the standard rule.

            * ``B -> N``
            * ``U -> C``
            * ``X -> G``
            * ``Z -> Q``
            * ``* -> <empty string>``
            * ``- -> <empty string>``

        Returns
        -----------
            Protein object

        """

        # If validation is needed...
        if validate:
            s = s.upper()
            if general_tools.is_valid_protein_sequence(s) is False:
                fixed = protfasta_utilities.convert_to_valid(s)

                if general_tools.is_valid_protein_sequence(fixed) is False:
                    raise sparrow_exceptions.SparrowException(
                        f"Invalid amino acid in {fixed}"
                    )

                self.__seq = fixed
            else:
                self.__seq = s.upper()

        else:
            self.__seq = s.upper()

        # all sequence parameters are initialized as unset class variables
        self.__aa_fracts = None
        self.__FCR = None
        self.__NCPR = None
        self.__hydrophobicity = None
        self.__aro = None
        self.__ali = None
        self.__polar = None
        self.__IDP_check = None
        self.__f_positive = None
        self.__f_negative = None
        self.__complexity = None
        self.__kappa = None
        self.__scd = None
        self.__shd = None
        self.__kappa_x = {}
        self.__linear_profiles = {}
        self.__molecular_weight = None
        self.__predictor_object = None
        self.__polymeric_object = None
        self.__plugin_object = None
        self.__elms = None

    # .................................................................
    #
    @property
    def molecular_weight(self):
        """
        Returns the molecular weight of the the protein.

        Returns
        ---------
        float
            The molecular weight

        """

        if self.__molecular_weight is None:
            self.__molecular_weight = physical_properties.calculate_molecular_weight(
                self.sequence
            )

        return self.__molecular_weight

    # .................................................................
    #
    @property
    def amino_acid_fractions(self):
        """
        Returns a dictionary where keys are each of the twenty standard
        amino acids and the values are the fraction of that amino acid
        in the sequence.

        Returns
        ---------
        dict
            Returns a 20-position dictionary that includes single-letter
            codes for each amino acid the corresponding fraction of those
            residues


        """

        if self.__aa_fracts is None:
            self.__aa_fracts = calculate_parameters.calculate_aa_fractions(self.__seq)

        return self.__aa_fracts

    # .................................................................
    #
    @property
    def FCR(self):
        """
        Returns the fraction of charged residues (FCR) in the sequence.
        Charged residues are Asp, Glu, Lys and Arg.

        Returns
        --------
        float
            Float between 0 and 1

        """
        if self.__FCR is None:
            self.__FCR = 0
            for i in data.amino_acids.CHARGE:
                self.__FCR = self.amino_acid_fractions[i] + self.__FCR
        return self.__FCR

    # .................................................................
    #
    @property
    def fraction_positive(self):
        """
        Returns the fraction of positively charged residues in the sequence.
        Positive residues are Arg and Lys (not His at physiological pH).

        Returns
        --------
        float
            Float between 0 and 1

        """
        if self.__f_positive is None:
            self.__f_positive = 0
            for i in data.amino_acids.POS:
                self.__f_positive = self.__f_positive + self.amino_acid_fractions[i]

        return self.__f_positive

    # .................................................................
    #
    @property
    def fraction_negative(self):
        """
        Returns the fraction of negatively charged residues in the
        sequence. Negative residues are Asp and Glu.

        Returns
        --------
        float
            Float between 0 and 1
        """
        if self.__f_negative is None:
            self.__f_negative = 0
            for i in data.amino_acids.NEG:
                self.__f_negative = self.__f_negative + self.amino_acid_fractions[i]

        return self.__f_negative

    # .................................................................
    #
    @property
    def NCPR(self):
        """
        Returns the net charge per residue of the sequence. Net
        charge is defined as (fraction positive) - (fraction negative)

        Returns
        --------
        float
            Float between -1 and +1
        """

        if self.__NCPR is None:
            self.__NCPR = self.fraction_positive - self.fraction_negative

        return self.__NCPR

    # .................................................................
    #
    @property
    def kappa(self):
        """
        Returns the charge segregation parameter kappa for the sequence.
        If kappa cannot be calculated (e.g. sequence is shorter than
        5 residues or a sequence with no charged residues) returns a -1.

        Note that kappa defaults to flattening kappa values above 1 to
        1, but this can be turned off with calculate_kappa_x() functio

        Returns
        --------
        float
            Float between 0 and 1, or -1
        """

        if self.__kappa is None:
            if len(self.sequence) < 6:
                self.__kappa = -1
            else:
                k5 = kappa.kappa_x(self.sequence, ["R", "K"], ["E", "D"], 5, 1)
                k6 = kappa.kappa_x(self.sequence, ["R", "K"], ["E", "D"], 6, 1)
                self.__kappa = (k5 + k6) / 2

        return self.__kappa

    @property
    def SCD(self):
        """
        Returns the default sequence charge decoration (SCD) parameter
        as defined by Sawle and Ghosh :cite:`sawle2015theoretical`.

        Returns
        --------
        float
            Returns a float that reports on the sequence charge decoration

        References
        ----------
        .. bibliography::
           :filter: key == "sawle2015theoretical"
        """
        if self.__scd is None:
            self.__scd = scd.compute_scd_x(
                self.sequence, group1=["E", "D"], group2=["R", "K"]
            )

        return self.__scd

    @property
    def SHD(self):
        """
        Returns the default sequence hydropathy decoration (SHD) parameter
        as defined by Zheng et al. :cite:`zheng2020hydropathy`.

        Returns
        --------
        float
            Returns a float that reports on the sequence hydropathy decoration

        References
        ----------
        .. bibliography::
           :filter: key == "zheng2020hydropathy"
        """

    # .................................................................
    #
    @property
    def fraction_aromatic(self):
        """
        Returns the fraction of aromatic residues in the sequence. Aromatic
        residues are Tyr, Phe, Trp.

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__aro is None:
            self.__aro = 0
            for i in data.amino_acids.ARO:
                self.__aro = self.amino_acid_fractions[i] + self.__aro

        return self.__aro

    # .................................................................
    #
    @property
    def fraction_aliphatic(self):
        """
        Returns the fraction of aliphatic residues in the sequence.
        Aliphatic residues are Ala, Leu, Ile, Met, Val.

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__ali is None:
            self.__ali = 0
            for i in data.amino_acids.ALI:
                self.__ali = self.amino_acid_fractions[i] + self.__ali

        return self.__ali

    # .................................................................
    #
    @property
    def fraction_polar(self):
        """
        Returns the fraction of polar residues in the sequence.
        Aliphatic residues are Gly, Ser, Thr, Gln, Asn, His.

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__polar is None:
            self.__polar = 0
            for i in data.amino_acids.POLAR:
                self.__polar = self.amino_acid_fractions[i] + self.__polar

        return self.__polar

    # .................................................................
    #
    @property
    def fraction_proline(self):
        """
        Returns the fraction of proline residues.

        Returns
        --------
        float
            Float between 0 and 1
        """
        return self.amino_acid_fractions["P"]

    # .................................................................
    #
    @property
    def hydrophobicity(self):
        """
        Returns the linear hydrophobicity from sequence
        using the Kyte-Doolitle scale.

        Returns
        ----------
        list
            List of values that correspond to per-residue
            hydrophobicity based on a given hydrophobicity scale
        """
        if self.__hydrophobicity is None:
            self.__hydrophobicity = calculate_parameters.calculate_hydrophobicity(
                self.__seq
            )

        return self.__hydrophobicity

    # .................................................................
    #
    @property
    def complexity(self):
        """
        Calculates the Wootton-Federhen complexity of a sequence (also called
        seg complexity, as this the theory used in the classic SEG algorithm.

        Returns
        ----------
        float
            Returns a float that corresponds to the compositional complexity
            associated with the passed sequence.
        """
        if self.__complexity is None:
            self.__complexity = calculate_parameters.calculate_seg_complexity(
                self.__seq
            )

        return self.__complexity

    # .................................................................
    #
    def compute_residue_fractions(self, residue_selector):
        """
        Compute the total fraction of specified residue types in the protein sequence.

        Parameters
        ----------
        residue_selector : list
            A list of one or more residue types (amino acid codes) to query
            in the sequence.

        Returns
        -------
        float
            The sum of fractions for all specified residue types. Returns 0.0
            if none of the specified residues are found in the sequence.

        Examples
        --------
        >>> protein.compute_residue_fractions(['A', 'G'])
        0.15
        >>> protein.compute_residue_fractions(['X', 'Z'])
        0.0
        """

        f = 0.0

        for i in residue_selector:
            if i in self.amino_acid_fractions:
                f = f + self.amino_acid_fractions[i]

        return f

    # .................................................................
    #
    def compute_kappa_x(self, group1, group2=None, window_size=6, flatten=True):
        """
        User-facing high-performance implementation for generic calculation of
        kappa_x. We use this for calculating real kappa (where group1 and
        group2 are ['E','D'] and ['R','K'], respectively but the function can
        be used to calculate arbitrary kappa-based patterning.

        NB1: kappa will return as -1 if

        1. the sequece is shorter than the windowsize
        2. There are no residues from either group1 or group2

        The function will raise an exception if the windowsize is < 2

        NB2: kappa is defined as comparing the ratio of delta with deltamax,
        where *in this implementation* deltamax refers to the delta associated
        with the most segregated sequece; e.g::

            (AAA)n-(XXX)m-(BBB)p

        Sometimes, when the charge asymmetry is VERY highly skewed, this most
        highly segregated sequence does not give the highest delta value, such
        that we can get a kappa greater than 1. This only occurs in situations
        where kappa is probably not a useful metric anyway (i.e 100x excess of
        one group residue vs. another). We recommend setting the 'flatten'
        keyword to True, which means kappa values over 1 will be flatteed to 1.

        NB3: this implementation differs very slightly from the canonical
        kappa reference implementation; it adds non-contributing 'wings' of
        the windowsize onto the N- and C-termini of the sequence. This
        means residue clusters at the end contribute to the overall sequence
        patterning as much as those in the middle, and also ensures we can
        analytically determine the deltamax sequence for arbitrary
        windowsizes.

        This both addresses a previous (subtle) limitation in kappa, but also
        buys a ~100x speedup compared to previous reference implementations.
        As a final note, I (Alex) wrote the original reference implementation
        in localCIDER, so feel comfortable criticising it's flaws!

        NB4: If no residues are provided in group2 then the function assumes
        all residues not defined in group1 are in group2 and the function
        becomes a binary patterning function instead of a ternary pattering
        function.

        Parameters
        -------------
        group1 : str
            Must be a string of valid amino acid one letter codes.
            This defines one set of residues for which patterning
            is computed against. If a second set is not provided,
            patterning is done via group1 vs. all other residues.

        group2 : str
            If provided, this defines the SECOND set of residues,
            such that patterning is done as residues in group1 vs.
            group2 in the background of everything else.

        window_size : int
            Size over which local sequence patterning will be
            calculated. Default = 6.

        flatten : bool
            Flag which, if set to True, means if kappa is above 1 the
            function will flatten the value to 1. Default = True.

        Returns
        -------------
        float
            Returns a value associated with the generalized kappa. If
            flatten is True this is guarenteed to be between 0 and 1
            unless it's -1 (see above). If flatten is set to False its
            VERY likeli this will be between 0 and 1 and if it's above
            1 the parameter is probably not useful to use.

        """

        for i in group1:
            if i not in amino_acids.VALID_AMINO_ACIDS:
                raise sparrow_exceptions.ProteinException(
                    f"Amino acid {i} (in group 1) is not a standard amino acid"
                )

            # make sure order is always consistent
            group1 = "".join(sorted(group1))

        # now deal with group 2
        if group2 is None:
            group2 = ""
        else:
            for i in group2:
                if i not in amino_acids.VALID_AMINO_ACIDS:
                    raise sparrow_exceptions.ProteinException(
                        f"Amino acid {i} (in group 2) is not a standard amino acid"
                    )

            # make sure order is always consistent
            group2 = "".join(sorted(group2))

        if flatten:
            kappa_x_name = group1 + "-" + group2 + str(window_size) + "flat"
        else:
            kappa_x_name = group1 + "-" + group2 + str(window_size)

        # after set up, calculate kappa_x
        if kappa_x_name not in self.__kappa_x:
            if flatten:
                self.__kappa_x[kappa_x_name] = kappa.kappa_x(
                    self.sequence, list(group1), list(group2), window_size, 1
                )
            else:
                self.__kappa_x[kappa_x_name] = kappa.kappa_x(
                    self.sequence, list(group1), list(group2), window_size, 0
                )

        return self.__kappa_x[kappa_x_name]

    # .................................................................
    #
    def compute_iwd(self, target_residues):
        """
        Returns the inverse weighted distance (IWD), a metric for
        residue clustering

        Parameters
        -------------
        target_residues : str or list
            One or more valid amino acid one-letter residue codes that define
            the target set for IWD clustering. This can be passed either as a
            joined string (for example ``"ILVAM"``) or as an iterable of
            residues (for example ``["I", "L", "V", "A", "M"]``).

        Returns
        --------
        float
            Float that is positive

        """
        residues = general_tools.normalize_residue_selector(
            target_residues,
            selector_name="target_residues",
            exception_cls=sparrow_exceptions.ProteinException,
            uppercase=True,
            require_nonempty=False,
            unique=False,
            sort_unique=False,
            return_type="str",
        )

        return iwd.calculate_average_inverse_distance_from_sequence(
            self.sequence, residues
        )

    # .................................................................
    #
    def compute_patch_fraction(
        self,
        residue_selector,
        interruption=2,
        min_target_count=4,
        adjacent_pair_pattern=None,
        min_adjacent_pair_count=0,
    ):
        """
        Returns the sequence fraction covered by residue patches.

        Parameters
        ----------
        residue_selector : str or list
            One or more amino acid one-letter residue codes defining patch hits.
        interruption : int, optional
            Maximum number of non-target residues bridged inside a candidate
            patch. Default is 2.
        min_target_count : int or None, optional
            Minimum number of target residues required for a bridged region to
            count as a patch. Default is 4. If set to ``None`` this filter is
            disabled.
        adjacent_pair_pattern : str or list, optional
            Optional adjacent residue motif that must occur in a bridged region
            (for example ``"RG"``). Default is None.
        min_adjacent_pair_count : int, optional
            Minimum number of occurrences of ``adjacent_pair_pattern`` required
            for a bridged region to count. Default is 0.

        Returns
        -------
        float
            Fraction of sequence positions covered by valid patch spans.
        """
        return patching.patch_fraction(
            self.sequence,
            residue_selector=residue_selector,
            interruption=interruption,
            min_target_count=min_target_count,
            adjacent_pair_pattern=adjacent_pair_pattern,
            min_adjacent_pair_count=min_adjacent_pair_count,
        )

    # .................................................................
    #
    def compute_rg_patch_fraction(self, interruption=2, min_adjacent_rg_pairs=2):
        """
        Returns the sequence fraction covered by RG motif-enriched patches.

        Parameters
        ----------
        interruption : int, optional
            Maximum number of non-R/G residues bridged inside a candidate RG
            patch. Default is 2.
        min_adjacent_rg_pairs : int, optional
            Minimum number of adjacent ``RG`` pairs required for a bridged
            region to count. Default is 2.

        Returns
        -------
        float
            Fraction of sequence positions covered by valid RG patch spans.
        """
        return self.compute_patch_fraction(
            residue_selector="RG",
            interruption=interruption,
            min_target_count=None,
            adjacent_pair_pattern="RG",
            min_adjacent_pair_count=min_adjacent_rg_pairs,
        )

    # .................................................................
    #
    def extract_feature_vector(
        self,
        patterning_config=None,
        composition_stats=None,
        use_default_composition_stats=True,
        include_raw=False,
        return_array=True,
        return_feature_names=False,
        backend=None,
        num_scrambles=None,
        blob_size=None,
        min_fraction=None,
        seed=None,
        fit_method=None,
    ):
        """
        Returns a grammar feature vector for the current sequence.

        This is a convenience wrapper over
        :func:`sparrow.sequence_analysis.grammar.compute_feature_vector`.

        This feature extraction interface is currently **alpha**. Breaking
        changes to arguments, defaults, and returned feature schemas may occur
        in future releases.

        This implementation is inspired by grammar-style analyses but is **not**
        an exact replica of the original NARDINI analysis pipeline.

        Parameters
        ----------
        patterning_config : GrammarPatterningConfig, optional
            Full patterning config for advanced control.
        backend : str, optional
            Override patterning backend (``"kappa_cython"`` or ``"iwd_combined"``).
        num_scrambles : int, optional
            Number of sequence scrambles used for patterning z-score estimation.
        blob_size : int, optional
            Patterning window size for kappa-style calculations.
        min_fraction : float, optional
            Minimum group fraction required to evaluate a patterning feature.
        seed : int, optional
            Random seed used for scramble generation.
        fit_method : str, optional
            Distribution fit mode (``"gamma_mle"`` or ``"moments"``).
        composition_stats : GrammarCompositionStats, optional
            Optional composition/patch background statistics used to add
            composition z-scores. The default uses sparrow's built-in human IDR composition background stats.
        use_default_composition_stats : bool, optional
            If True and ``composition_stats`` is None, use Sparrow's built-in
            human-IDR composition background stats. Default True.
        include_raw : bool, optional
            Include raw feature block (``raw::`` keys). Default False.
        return_array : bool, optional
            If True, return a NumPy array (``np.float32``) instead of an OrderedDict.
            Default True.
        return_feature_names : bool, optional
            If True and ``return_array=True``, also return an ordered tuple of
            feature names.

        Returns
        -------
        numpy.ndarray or collections.OrderedDict
            Feature vector. Returns an array by default.
        """
        # local import to prevent the circular import between this module and
        # sparrow.sequence_analysis.grammar which uses the Protein class
        # to calculate many different sequence features.
        from sparrow.sequence_analysis import grammar

        return grammar.compute_feature_vector(
            sequence_or_protein=self,
            patterning_config=patterning_config,
            composition_stats=composition_stats,
            use_default_composition_stats=use_default_composition_stats,
            include_raw=include_raw,
            return_array=return_array,
            return_feature_names=return_feature_names,
            backend=backend,
            num_scrambles=num_scrambles,
            blob_size=blob_size,
            min_fraction=min_fraction,
            seed=seed,
            fit_method=fit_method,
        )

    # .................................................................
    #
    def compute_SCD_x(self, group1, group2):
        """
        Function that computes the sequence charge decoration (SCD)
        parameter of Sawle and Ghosh. This is an alternative sequence
        patterning parameter which we provide here generalized such
        that it determines the patterning between any two groups of
        residues.

        Parameters
        --------------

        group1 : str or list
            Collection of amino acids to be used for defining "negatively
            charged" residues.

        group2 : str or list
            Collection of amino acids to be used for defining "positively
            charged" residues.

        Returns
        -----------
        float
            Returns the custom sequence charge decoration.

        See also
        ---------
        sparrow.protein.scd

        References
        ----------
        .. bibliography::
           :filter: key == "sawle2015theoretical"
        """

        return scd.compute_scd_x(self.sequence, group1=group1, group2=group2)

    # .................................................................
    #
    def compute_SHD_custom(self, hydro_dict):
        """
        Function takes in a sequence and returns Sequence
        Hydropathy Decoration (SHD), IE. patterning of hydrophobic
        residues in the sequence. This is computed as define in ref 1

        To define the hydrophobicity values used the user should pass
        a hydrophobicity dictionary (hydro_dict) which maps amino
        acid residues to hydrophobicity scores.

        Parameters
        --------------

        hydro : dict
            Dictionary that maps amino acid to hydrophobicity score.
            Note that every amino acid in the sequence must exist in the
            hydro dict or the function raise an exception.

        Returns
        -----------
        float
            Returns the customized sequence hydropathy decoration

        See also
        ---------
        sparrow.protein.scd

        References
        ----------
        .. bibliography::
           :filter: key == "zheng2020hydropathy"
        """

        return scd.compute_shd(self.sequence, hydro_dict=hydro_dict)

    # .................................................................
    #
    def compute_iwd_charged_weighted(self, charge=None):
        """
        Returns the weighted inverse weighted distance (IWD) for either
        positive or negative residues in the sequence. This is a metric
        for residue clustering weighted by the NCPR of each target
        residue.

        Parameters
        -------------

        charge : ['-','+']
            Pass '-' to quantify the clustering of negitive residues.
            Pass '+' to quantify the clustering of positive residues.

        Returns
        --------
        float
            Float that is positive
        """

        # ensure valid charge is passed
        if charge not in ["-", "+"]:
            raise sparrow_exceptions.ProteinException(
                f'Passed charge {charge} is not a valid option. Pass "-" for negitive residues and "+" for positive residues.'
            )

        # calculate or retrieve mask of NCPR for sequence
        if "NCPR-8-extend-ends" not in self.__linear_profiles:
            self.__linear_profiles["NCPR-8-extend-ends"] = (
                track_tools.predefined_linear_track(
                    self.__seq, "NCPR", 8, "extend-ends", None
                )
            )

        linear_NCPR = self.__linear_profiles["NCPR-8-extend-ends"]

        return iwd.calculate_average_inverse_distance_charge(
            linear_NCPR, self.sequence, charge
        )

    # .................................................................
    #
    def compute_bivariate_iwd_charged_weighted(self):
        """
        Returns the a weighted bivariate inverse weighted distance (IWD) for
        between Possitive and Negative residues in the sequence, a metric for
        residue clustering weighted by the difference in NCPR of the target residues.

        Returns
        --------
        float
            Float that is positive
        """

        # calculate or retrieve mask of NCPR for sequence
        if "NCPR-8-extend-ends" not in self.__linear_profiles:
            self.__linear_profiles["NCPR-8-extend-ends"] = (
                track_tools.predefined_linear_track(
                    self.__seq, "NCPR", 8, "extend-ends", None
                )
            )

        linear_NCPR = self.__linear_profiles["NCPR-8-extend-ends"]

        return iwd.calculate_average_bivariate_inverse_distance_charge(
            linear_NCPR, self.sequence
        )

    ## .................................................................
    ##
    def generate_phosphoisoforms(self, mode="all", phospho_rate=1, phosphosites=None):
        """
        Calls sequence_analysis.phospho_isoforms module to get a list
        of possible phosphoisoforms sequences. See module header for more documentation.

        Phosphosites are replaced with the phosphomimetic 'E', enabling approximate calculation
        of charge based sequence features with the presence of a phosphorylated residues.

        Parameters
        ----------
        sequence : str
            Valid amino acid sequence

        mode : str, optional
            Defition for how the phosphosites should be determined, by default "all"

            'all'       : Assumes all S/T/Y residues are potential phosphosites

            'predict'   : Leverages PARROT trained predictors via _predict_all_phosphosites
                            to predict phosphorylated sites based on sequence.

            'custom'    : uses the 'phosphosites' parameter as indices for phosphosites.

        phospho_rate : int, optional
            Value between 0 and 1 which defines the maximum percent of phosphosites
            can be 'phosphorylated' a each sequence, by default 1 (IE all sites can be
            phosphorylated)

        phosphosites : list, optional
            Custom list of indices for valid phosphosite positions, by default None

        Returns
        -------
        list
            list of sequences for the possible phosphoisoforms based off the selected method.
            Phosphorylatable amino acids are replaced with 'E'.
        """
        return phospho_isoforms.get_phosphoisoforms(
            self.sequence,
            mode=mode,
            phospho_rate=phospho_rate,
            phosphosites=phosphosites,
        )

    # .................................................................
    #
    def linear_sequence_profile(
        self, mode, window_size=8, end_mode="extend-ends", smooth=None
    ):
        """
        Function that returns a vectorized representation of local composition/sequence properties, as defined
        by the passed 'mode', which acts as a selector toggle for a large set of pre-defined analyses types.

        Parameters
        ----------
        mode : str
            Selector for the type of analysis to perform:

            * ``'FCR'`` : Fraction of charged residues
            * ``'NCPR'`` : Net charge per residue
            * ``'aromatic'`` : Fraction of aromatic residues
            * ``'aliphatic'`` : Fraction of aliphatic residues
            * ``'polar'`` : Fraction of polar residues
            * ``'proline'`` : Fraction of proline residues
            * ``'positive'`` : Fraction of positive residues
            * ``'negative'`` : Fraction of negative residues
            * ``'hydrophobicity'`` : Linear hydrophobicity (Kyte-Doolitle)
            * ``'seg-complexity'`` : Linear complexity
            * ``'kappa'`` : Linear charge patterning

        window_size : int
            Number of residues over which local sequence properties are calculated. A window stepsize of 1
            is always used.

        end_mode : str
            Selector that defines how ends are dealt with. Default is ``'extend-ends'``.

            * ``'extend-ends'`` : The leading/lagging track values are copied from the first and last values.
            * ``''`` : Empty string means they're ignored.
            * ``'zero-ends'`` : Leading/lagging track values are set to zero.

        smooth : int or None
            Selector which allows you to smooth the data over a windowsize. Note window
            must be an odd number (applies a savgol_filter with a 3rd order polynomial
            which requires an odd number).

        Returns
        -------
        list
            Returns a list with values that correspond to the passed mode.
        """

        utilities.validate_keyword_option(
            mode,
            [
                "FCR",
                "NCPR",
                "aromatic",
                "aliphatic",
                "polar",
                "proline",
                "positive",
                "negative",
                "hydrophobicity",
                "seg-complexity",
                "kappa",
            ],
            "mode",
        )

        if smooth is not None:
            name = "%s-%i-%s-%i" % (mode, window_size, end_mode, smooth)
        else:
            name = "%s-%i-%s" % (mode, window_size, end_mode)

        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.predefined_linear_track(
                self.__seq, mode, window_size, end_mode, smooth
            )

        return self.__linear_profiles[name]

    # .................................................................
    #
    def linear_composition_profile(
        self, composition_list, window_size=8, end_mode="extend-ends", smooth=None
    ):
        """
        Function that returns a vectorized representation of local composition/sequence properties, as defined
        by the set of one or more residues passed in composition_list.

        Parameters
        ------------

        composition_list : list
            List where each element should be a valid amino acid

        window_size : int
            Number of residues over which local sequence properties are calculated. A window stepsize of 1
            is always used

        end_mode : str
            Selector that defines how ends are dealt with. Empty string means nothing is
            done, but extend-ends and zero-ends ensure the track length equals the sequence
            length which can often be useful. Default is 'extend-ends'.

            'extend-ends'   |    The leading/lagging track values are copied from
                                 the first and last and values.

            ''              |    Empty string means they're ignored,

            'zero-ends'     |    Means leading/lagging track values are set to zero.

        smooth : int or None
            Selector which allows you to smooth the data over a windowsize. Note window
            must be an odd number (applies a savgol_filter with a 3rd order polynomial
            which requires an odd number).


        Returns
        ----------
        list
            Returns a list with values that correspond to the passed mode

        """

        utilities.validate_keyword_option(
            end_mode, ["extend-ends", "zero-ends", ""], "end_mode"
        )

        # we sort the composition list to unify how it is saved for memoization
        try:
            composition_list = list(set(composition_list))
            composition_list.sort()

        except AttributeError:
            raise sparrow_exceptions.ProteinException(
                "Unable to sort composition_list (%s) - this should be a list"
                % (str(composition_list))
            )

        name = (
            "-".join(composition_list)
            + "-window_size=%i" % (window_size)
            + "-end_mode=%s" % (end_mode)
            + "smooth=%s" % (smooth)
        )

        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.linear_track_composition(
                self.__seq,
                composition_list,
                window_size=window_size,
                end_mode=end_mode,
                smooth=smooth,
            )

        return self.__linear_profiles[name]

    # .................................................................
    #
    def low_complexity_domains(self, mode="holt", **kwargs):
        """Extract low complexity domains (LCDs) from the sequence.

        Parameters
        ----------
        mode : {'holt'}
            Extraction method. Only ``'holt'`` currently supported (Gutierrez et al. method).
        **kwargs
            Passed through to :func:`sparrow.sequence_analysis.sequence_complexity.low_complexity_domains_holt`.
            Common options:

            ``residue_selector`` : str
                One or more one-letter amino acid codes (e.g. ``'Q'`` or ``'ED'``).
            ``minimum_length`` : int, default 15
                Minimum allowed LCD length.
            ``max_interruption`` : int, default 5
                Maximum number of consecutive residues NOT in ``residue_selector`` permitted
                inside an LCD (Gutierrez et al. used 17).
            ``fractional_threshold`` : float, default 0.25
                Minimum fraction (0-1) of residues from ``residue_selector`` required in the LCD.

        Returns
        -------
        list[list]
            Each LCD represented as ``[sequence, start, end]`` where ``start`` is 0-indexed
            and ``end`` is exclusive (``sequence[start:end]`` equals the LCD substring).

        Notes
        -----
        Only the Gutierrez et al. style extraction (``mode='holt'``) is implemented at present.

        Examples
        --------
        >>> p.low_complexity_domains(mode='holt', residue_selector='Q', minimum_length=10)  # doctest: +SKIP
        [['QQQQQQQQQQ', 5, 15]]
        """
        # utilities.validate_keyword_option(mode, ['holt', 'holt-permissive'], 'mode')
        utilities.validate_keyword_option(mode, ["holt"], "mode")
        if mode == "holt":
            return sequence_complexity.low_complexity_domains_holt(
                self.sequence, **kwargs
            )
        if mode == "holt-permissive":
            return sequence_complexity.low_complexity_domains_holt_permissive(
                self.sequence, **kwargs
            )

    def show_sequence(
        self,
        blocksize=10,
        newline=50,
        fontsize=14,
        font_family="Courier",
        colors={},
        header=None,
        bold_positions=[],
        bold_residues=[],
        opaque_positions=[],
        return_raw_string=False,
        warnings=True,
    ):
        """
        Function that generates an HTML colored string that either renders in the browser or returns the
        html string. Contains various customizable components.

        Parameters
        -------------

        blocksize : int
            Defines how big blocks of residues are. Blocks are equal to blocksize or the newline parameter, whicever is smaller.
            Default=10. If set to -1 uses length of the sequence.

        newline : int
            Defines how many residues are shown before a newline is printed. Default is 50. If set to -1 uses the length of
            the sequence.

        fontsize : int
            Fontsize used. Default is 14

        font_family : str
            Which font family (from HTML fonts) is used. Using a non-monospace font makes no sense as columns will be
            unaligned. Default is Courier.

        colors : dict
            Dictionary that allows overiding of default color scheme. Should be of format key-value as 'residue'-'color' where
            residue is a residue in the string and color is a valid HTML color (which can be a Hexcode, standard HTML color name).
            Note that this also lets you define colors for non-standard amino acids should these be useful. Default is an empty
            dictionary. Note also that the standard amino acid colorings are defined at sparrow.data.amino_acids.AA_COLOR

        header : str
            If provided, this is a string that provides a FASTA-style header (with a leading carrett included). Default None.

        bold_positions : list
            List of positions (indexing from 1 onwards) which will be bolded. Useful for highlighting specific regions. Note that this
            defines individual residues so (for example) to bold residues 10 to 15 would require bold_positions=[10,11,12,13,14,15].
            Default is an empty list.

        bold_residues : list
            List of residue types that can be bolded. Useful for highlighting specific residue groups.  Default is an empty list.

        opaque_positions : list
            List of positions (indexing from 1 onwards) which will be grey and slighlty opaque. Useful for highlighting specific regions.
            Note that this defines individual residues so (for example) to bold residues 10 to 15 would require
            bold_positions=[10,11,12,13,14,15]. Default is an empty list.

        return_raw_string : bool
            If set to true, the function returns the actual raw HTML string, as opposed to an in-notebook rendering.
            Default is False

        warnings : bool
            If set to True, will print warnings if an invalid amino acid is encountered. Default is True.

        Returns
        ----------
        None or str
            If return_raw_string is set to True then an HTML-compatible string is returned.


        Raises
        -------
        sparrow.sparrow_exceptions.SparrowException
            Raises a sparrow exception if invalid input is provided (within reason).

        """

        r_val = sequence_visuals.show_sequence(
            self.sequence,
            blocksize=blocksize,
            newline=newline,
            fontsize=fontsize,
            font_family=font_family,
            colors=colors,
            header=header,
            bold_positions=bold_positions,
            bold_residues=bold_residues,
            opaque_positions=opaque_positions,
            return_raw_string=return_raw_string,
            warnings=warnings,
        )

        if return_raw_string:
            return r_val

    @property
    def plugin(self):
        """
        Returns a ``sparrow.sequence_analysis.plugins.PluginManager`` object which
        provides programmatic access to the various different plugins implemented
        in sparrow.
        """
        if self.__plugin_object is None:
            from sparrow.sequence_analysis.plugins import PluginManager  # local import

            self.__plugin_object = PluginManager(self)
        return self.__plugin_object

    @property
    def predictor(self):
        """
        Returns a ``sparrow.predictors.Predictor`` object which provides programatic
        access to the various different sequence-based predictors implemented in
        sparrow.

        Note that each predictor performs necessary imports at runtime on the first
        execution for the first protein, minimizing unnecessary overhead.

        Currently available predictors include:

            * disorder : per-residue disorder prediction
            * dssp : per-residue DSSP score (0, 1, or 2)
            * nes : nuclear export signal
            * nis : nuclear import signal
            * phosphorylation
            * pscore
            * tad
            * mitochondrial targeting
            * rg : radius of gyration
            * transmembrane_region : binary classification of transmembrane regions

        """
        if self.__predictor_object is None:
            from sparrow.predictors import Predictor  # local import

            self.__predictor_object = Predictor(self)
        return self.__predictor_object

    @property
    def polymeric(self):
        """
        Returns a ``sparrow.polymer.Polymeric`` object which provides programatic
        access to the various predicted polymer properties for the sequence.

        Note that many of these properties assume the sequence behaves as an
        intrinsically disordered or unfolded polypeptide.

        """
        if self.__polymeric_object is None:
            from sparrow.polymer import Polymeric  # local import

            self.__polymeric_object = Polymeric(self)
        return self.__polymeric_object

    @property
    def elms(self):
        """Returns a list of NamedTuples containing each of the
        elm annotations for the given sequence.

        Returns
        -------
        List[NamedTuple]
            A list of NamedTuples containing all possible elms in a given sequence.

        """
        if self.__elms is None:
            self.__elms = elm.find_all_elms(self.sequence)
        return self.__elms

    @property
    def sequence(self):
        """Returns a string representation of the protein sequence.

        Returns
        -------
        str
            The protein sequence.
        """
        return self.__seq

    def __len__(self):
        """Returns the length of the protein sequence.

        Returns
        -------
        int
            The length of the protein sequence.
        """
        return len(self.__seq)

    def __repr__(self):
        s = self.__seq[0:5]
        if len(s) < 5:
            s = s + "." * (5 - len(s))
        return f"Protein|L = {len(self)}]|{s}..."
