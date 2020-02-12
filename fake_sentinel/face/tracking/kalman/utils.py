import warnings
import numpy as np
from scipy.stats import multivariate_normal


# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        'You are using a version of SciPy that does not support the '\
        'allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). '\
        'Future versions of FilterPy will require a version of SciPy that '\
        'implements this keyword',
        DeprecationWarning)
    _support_singular = False


def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.
    Wrapper for older versions of scipy.multivariate_normal.logpdf which
    don't support support the allow_singular keyword prior to verion 0.15.0.
    If it is not supported, and cov is singular or not PSD you may get
    an exception.
    `x` and `mean` may be column vectors, row vectors, or lists.
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)


def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


def pretty_str(label, arr):
    """
    Generates a pretty printed NumPy array with an assignment. Optionally
    transposes column vectors so they are drawn on one line. Strictly speaking
    arr can be any time convertible by `str(arr)`, but the output may not
    be what you want if the type of the variable is not a scalar or an
    ndarray.
    Examples
    --------
    >>> pprint('cov', np.array([[4., .1], [.1, 5]]))
    cov = [[4.  0.1]
           [0.1 5. ]]
    >>> print(pretty_str('x', np.array([[1], [2], [3]])))
    x = [[1 2 3]].T
    """

    def is_col(a):
        """ return true if a is a column vector"""
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except (AttributeError, IndexError):
            return False

    if label is None:
        label = ''

    if label:
        label += ' = '

    if is_col(arr):
        return label + str(arr.T).replace('\n', '') + '.T'

    rows = str(arr).split('\n')
    if not rows:
        return ''

    s = label + rows[0]
    pad = ' ' * len(label)
    for line in rows[1:]:
        s = s + '\n' + pad + line

    return s
