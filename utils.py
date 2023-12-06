import numpy as np
import numpy.typing as npt


def print_final_msg(times: list[float], errors: list[float], init_time: float, i: int):
    avg_time_per_iter = np.mean(times)
    total_time = init_time + np.sum(times)
    print(f"\nFinal relative error: {100 * errors[-1]}%, after {i + 1} iterations.")
    print(f"Initialization time: {init_time:3f} secs")
    print(f"Mean time per iteration: {avg_time_per_iter:3f} secs")
    print(f"Total time: {total_time:3f} secs\n")


def compute_abs_error(
    Theta: npt.NDArray[np.float_], X: npt.NDArray[np.float_]
) -> float:
    """Compute Frobenius norm of the difference between Theta and X"""
    return np.linalg.norm(np.maximum(0, Theta) - X, ord="fro")
