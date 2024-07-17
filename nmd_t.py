import numpy as np
import numpy.typing as npt
from typing import Tuple
from time import perf_counter
from utils import compute_abs_error, print_final_msg
from scipy.sparse.linalg import svds


def nmd_t(
    X: npt.NDArray[np.float_],
    r: int,
    W0: npt.NDArray[np.float_] = None,
    H0: npt.NDArray[np.float_] = None,
    beta: float = 0.95,
    regularization_lambda: float = 0.0001,
    max_iters: int = 1000,
    tol: float = 1e-4,
    tol_over_10iters: float = 1e-5,
    verbose: bool = True,
) -> Tuple[npt.NDArray[np.float_], list[float], int, list[float]]:
    """NMD using three-block alternating minimization with Tikhonov regularization

    Direct port of the matlab version: https://github.com/nothing2wang/NMD-TM

    Adaption of 3B-NMD, adding Tikhonov regularization on W and H as well as
    momentum steps on W and H with a small negative momentum parameter.

    Args:
        X (npt.NDArray[np.float_]): (m, n) sparse non-negative matrix
        r (int): approximation rank
        W0 (npt.NDArray[np.float_], optional): initial W.
            Defaults to np.random.randn(m, r) if none is provided.
        H0 (npt.NDArray[np.float_], optional): initial H.
            Defaults to np.random.randn(r, n) if none is provided.
        beta (float, optional): momentum parameter. Defaults to 0.95.
        regularization_lambda (float, optional): regularization parameter. Defaults to 0.0001.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional):  stopping criterion on the relative error:
            ||X-max(0,WH)||/||X|| < tol. Defaults to 1e-4.
        tol_over_10iters (float, optional): stopping criterion tolerance on 10
            successive errors: abs(errors[i] - errors[i-10]) < tol_err.
            Defaults to 1e-5.
        verbose (bool, optional): print information to console. Defaults to True.

    Returns:
        (npt.NDArray[np.float_], list[float], int, list[float]):
            Theta, errors_relative, number of iterations, times
    """

    if np.any(X < 0):
        raise ValueError("X must be non-negative.")

    start_time_init = perf_counter()

    beta2 = beta - 1.0

    m, n = X.shape
    W0 = np.random.randn(m, r) if W0 is None else W0
    H0 = np.random.randn(r, n) if H0 is None else H0

    norm_X = np.linalg.norm(X, "fro")
    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    Z = np.zeros((m, n))
    Z[x_is_pos] = X[x_is_pos]

    W, H = W0, H0
    Theta = W @ H
    W_old, H_old = W0.copy(), H0.copy()
    Z_old, Theta_old = Z.copy(), Theta.copy()

    errors = [compute_abs_error(Theta, X) / norm_X]

    if verbose:
        print(
            "Running NMD-T, evolution of [iteration number : relative error in %] - time per iteration"
        )

    initialization_time = perf_counter() - start_time_init
    times = []
    for i in range(0, max_iters):
        start_time_iteration = perf_counter()

        Z = np.minimum(0.0, Theta * x_is_zero)
        Z += X * x_is_pos
        Z = (1 + beta) * Z - beta * Z_old

        # Update of W
        W = np.linalg.lstsq(
            regularization_lambda * np.eye(r) + H_old @ H_old.T, H_old @ Z.T
        )[0].T
        W = (1 + beta2) * W - beta2 * W_old

        # Update of H
        H = np.linalg.lstsq(regularization_lambda * np.eye(r) + W.T @ W, W.T @ Z)[0]
        H = (1 + beta2) * H - beta2 * H_old
        Theta = W @ H

        errors.append(compute_abs_error(Theta, X) / norm_X)
        if errors[-1] < tol:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(f"\nConverged: ||X-max(0,WH)||/||X|| < {tol}")
            break

        if i >= 10 and abs(errors[-1] - errors[-11]) < tol_over_10iters:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(
                    f"\nConverged: abs(rel. err.(i) - rel. err.(i-10)) < {tol_over_10iters}"
                )
            break

        if i < max_iters - 1:
            Theta *= 1.0 + beta
            Theta -= beta * Theta_old

        Z_old, Theta_old, W_old, H_old = Z.copy(), Theta.copy(), W.copy(), H.copy()

        times.append(perf_counter() - start_time_iteration)

        if verbose:
            print(f"[{i} : {(100 * errors[-1]):5f}] - {times[-1]:3f} secs")

    if verbose:
        print_final_msg(times, errors, initialization_time, i)

    return Theta, W, H, errors, i + 1, times
