import numpy as np
import numpy.typing as npt
from typing import Tuple
from time import perf_counter
from utils import compute_abs_error, print_final_msg
from scipy.sparse.linalg import svds


def a_nmd(
    X: npt.NDArray[np.float_],
    r: int,
    Theta0: npt.NDArray[np.float_] = None,
    beta: float = 0.9,
    eta: float = 0.4,
    gamma: float = 1.05,
    gamma_bar: float = 1.1,
    max_iters: int = 1000,
    tol: float = 1.0e-4,
    tol_over_10iters: float = 1.0e-5,
    verbose: bool = True,
) -> Tuple[npt.NDArray[np.float_], list[float], int, list[float]]:
    """Aggressive Momentum NMD (A-NMD)

    Direct port of the matlab version: https://gitlab.com/ngillis/ReLU-NMD/

    Args:
        X (npt.NDArray[np.float_]): (m, n) sparse non-negative matrix
        r (int): approximation rank
        Theta0 (npt.NDArray[np.float_]): initial Theta. Defaults to np.random.randn(m, n) if none is provided.
        beta (float, optional): initial momentum parameter. Defaults to 0.9.
        eta (float, optional): hyperparameter. Defaults to 0.4.
        gamma (float, optional): hyperparameter. Defaults to 1.05.
        gamma_bar (float, optional): hyperparameter. Defaults to 1.1.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional):  stopping criterion on the relative error: ||X-max(0,WH)||/||X|| < tol. Defaults to 1e-4.
        tol_over_10iters (float, optional): stopping criterion tolerance on 10 successive errors: abs(errors[i] - errors[i-10]) < tol_err. Defaults to 1e-5.
        verbose (bool, optional): print information to console. Defaults to True.

    Returns:
        (npt.NDArray[np.float_], list[float], int, list[float]): Theta, errors_relative, number of iterations, times
    """
    if np.any(X < 0):
        raise ValueError("X must be non-negative.")

    ## Code different than paper
    # assert eta > 1.0
    # assert eta > gamma
    # assert gamma > gamma_bar

    start_time_init = perf_counter()
    m, n = X.shape
    Theta0 = np.random.randn(m, n) if Theta0 is None else Theta0
    beta_bar = 1.0
    beta_history = [beta]

    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    Z0 = np.zeros((m, n))
    Z0[x_is_pos] = X[x_is_pos]

    Z = Z0
    Theta = Theta0
    Z_old = Z0.copy()
    Theta_old = Theta0.copy()

    norm_X = np.linalg.norm(X, ord="fro")
    errors_relative = [compute_abs_error(Theta, X) / norm_X]

    if verbose:
        print(
            "Running A-NMD, evolution of [iteration number : relative error in %] - time per iteration"
        )

    initialization_time = perf_counter() - start_time_init
    times = []
    for i in range(max_iters):
        start_time_iteration = perf_counter()

        Z = np.minimum(0.0, Theta * x_is_zero)
        Z += X * x_is_pos
        Z += beta * (Z - Z_old)

        U, d, Vt = svds(Z, r)
        D = np.diag(d)
        Theta = U @ D @ Vt

        errors_relative.append(compute_abs_error(Theta, X) / norm_X)

        if errors_relative[-1] < tol:
            if verbose:
                print(f"\nConverged: ||X-max(0,WH)||/||X|| < {tol}")
            break

        if (
            i >= 10
            and abs(errors_relative[-1] - errors_relative[-11]) < tol_over_10iters
        ):
            if verbose:
                print(
                    f"\nConverged: abs(rel. err.(i) - rel. err.(i-10)) < {tol_over_10iters}"
                )
            break

        if i < max_iters - 1:
            Theta += beta * (Theta - Theta_old)

        if i > 1:
            if compute_abs_error(Theta, X) < compute_abs_error(Theta_old, X):
                beta = min(beta_bar, gamma * beta)
                beta_bar = min(1, gamma_bar * beta)  # in paper: gamma_bar * beta_bar
                beta_history.append(beta)

                Z_old = Z.copy()
                Theta_old = Theta.copy()
            else:
                beta *= eta
                beta_history.append(beta)
                beta_bar = beta_history[i - 2]

                Z = Z_old.copy()
                Theta = Theta_old.copy()

        times.append(perf_counter() - start_time_iteration)

        if verbose:
            print(f"[{i} : {(100 * errors_relative[-1]):5f}] - {times[-1]:3f} secs")

    if verbose:
        print_final_msg(times, errors_relative, initialization_time, i)

    return Theta, errors_relative, i + 1, times
