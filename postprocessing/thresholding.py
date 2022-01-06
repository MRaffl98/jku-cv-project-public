import numpy as np

def threshold(losses, mode='q95'):
    """
    @param losses: array_like of reconstruction errors
    @param mode: str (thresholding strategy)
        - 'q{quantile}' binarizes w.r.t the given quantile. (e.g. 'q95')
    @returns: zero-one array with same shape as losses
    """
    if mode.startswith('q'):
        q = float(mode[1:]) / 100
        quantile = np.quantile(losses, q)
        return np.where(losses >= quantile, 1, 0)


