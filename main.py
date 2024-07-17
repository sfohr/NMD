from a_nmd import a_nmd
from nmd_3b import nmd_3b
from nmd_t import nmd_t
from nuclear_norm_init import nuclear_norm_init
import numpy.typing as npt
import numpy as np


def generate_sparse_matrix(m: int, n: int, r: int, c: float = 0.0) -> npt.NDArray:
    """Generates a (m, n) matrix X with rank r

    X is generated as min(0, W1@H1), where W1 is of size (m, r)
    and H1 of size (r, n), entries in W1 and H1 are standard normal, therefore,
    on average, 50% of the entries of X are sparse if c=0.
    If c > 0, then the resulting matrix will have more than 50% sparsity.

    Args:
        m (int): Number of rows in X
        n (int): Number of columns in X
        r (int): Desired rank
        c (float): sparsity parameter 0 <= c <= 1.7

    Returns:
        np.ndarray: A sparse rank r matrix of shape (m, n) entries.
    """
    W = np.random.randn(m, r) - c
    H = np.random.randn(r, n) + c
    return np.maximum(0, W @ H)


if __name__ == "__main__":
    m = 200
    n = m
    r = 8
    c = 0.8  # with c = 0.8 & r=8, matrix will have ~90% negative entries
    X = generate_sparse_matrix(m, n, r, c)
    print(f"Sparsity: {np.sum(X == 0) / X.size}")

    # initialization
    W0, H0 = nuclear_norm_init(X, m, n, r=r)
    Theta0 = W0 @ H0

    # Aggressive momentum NMD
    theta_a, loss_a, i_a, times_a = a_nmd(X, r=r, Theta0=Theta0)

    # 3-block NMD
    theta_3b, loss_3b, i_3b, times_3b = nmd_3b(X, r=r, W0=W0, H0=H0)

    # NMD-T
    theta_t, W_t, H_t, loss_t, i_t, times_t = nmd_t(X, r=r, W0=W0, H0=H0)
