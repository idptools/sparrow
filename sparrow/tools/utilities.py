from sparrow.sparrow_exceptions import SparrowException

def validate_keyword_option(keyword, allowed_vals, keyword_name, error_message=None):
    """
    Helper function that checks a passed keyword is only one of a set of possible
    valid keywords

    Parameters
    -----------
    keyword : str
        The actual passed keyword value

    allowed_vals : list of str
        A list of possible keywords

    keyword_name : str
        the name of the keyword as the user would select it in the function call

    error_message : str
        Allows the user to pass a custom error message


    Returns
    --------
    None

        No return value, but raises ctexceptions.CTException if ``keyword `` is not
        found in the allowed_vals list
           
    """


    if keyword not in allowed_vals:
        if error_message is None:
            raise SparrowException(f'Keyword {keyword_name} passed value [{keyword}], but this is not valid.\nMust be one of: {str(allowed_vals)}')
