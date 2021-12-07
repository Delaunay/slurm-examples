__descr__ = "seeddescription"
__version__ = "seedversion"
__license__ = "seedlicense"
__author__ = u"seedauthor"
__author_email__ = "seed@email"
__copyright__ = u"seedcopyright seedauthor"
__url__ = "seedurl"


def my_function(a: int, b: int) -> int:
    """Add two numbers together

    Parameters
    ----------
    a: int
        first integer

    b: int
        second integer

    Raises
    ------
    value errror if a == 0

    Examples
    --------

    >>> my_function(1, 2)
    3
    >>> my_function(0, 2)
    Traceback (most recent call last):
      ...
    ValueError

    """
    if a == 0:
        raise ValueError()

    return a + b
