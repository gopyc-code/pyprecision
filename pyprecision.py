from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import t


def _normalize_base10(
    value: Union[float, np.floating]
) -> Tuple[np.float64, np.float64, int]:
    """
    Convert a scalar to sign, mantissa and exponent in base-10 normalized form:
    value = sign * mantissa * 10**exponent, where 1 <= mantissa < 10 for value != 0.
    Small floating-point artifacts are suppressed by rounding the mantissa to 12 decimals.
    All intermediate numeric types are numpy.float64.
    """
    v = np.asarray(value, dtype=np.float64)
    if v == 0.0:
        return np.float64(1.0), np.float64(0.0), 0

    sign = np.float64(-1.0) if v < 0 else np.float64(1.0)
    abs_val = np.abs(v)
    exponent = int(np.floor(np.log10(abs_val)))
    mantissa = abs_val / (10.0 ** exponent)
    mantissa = np.round(mantissa, 12).astype(np.float64)
    if mantissa >= np.float64(10.0):
        mantissa = mantissa / np.float64(10.0)
        exponent += 1
    return sign, mantissa, exponent


def regop_val(
    value: Union[float, int, np.floating], 
    mode: str = "simple"
) -> Tuple[np.float64, int]:
    """
    Round Errors and Get Orders of Precision for Values (REGOP Value)

    Modes:
        'gost'   - GOST-like: keep two significant digits if the first significant
                   digit is 1, 2 or 3; otherwise keep one.
        'simple' - Simplified: keep two significant digits when mantissa 1.0 <= M < 1.5,
                   otherwise keep one.

    The function uses base-10 normalization (mantissa in [1,10)) and performs the
    mantissa rounding with numpy double precision.
    """
    v = np.asarray(value, dtype=np.float64)
    if v == 0.0:
        return np.float64(0.0), 100

    if mode not in ("gost", "simple"):
        raise ValueError("mode must be 'gost' or 'simple'")

    sign, mantissa, exponent = _normalize_base10(v)

    if mode == "gost":
        first_digit = int(mantissa)  # 1..9
        sig_figs = 2 if first_digit in (1, 2, 3) else 1
    else:
        sig_figs = 2 if (mantissa >= np.float64(1.0) and mantissa < np.float64(1.5)) else 1

    if sig_figs == 1:
        rounded_m = np.round(mantissa).astype(np.float64)
    else:
        rounded_m = np.round(mantissa, 1).astype(np.float64)

    if rounded_m >= np.float64(10.0):
        rounded_m = rounded_m / np.float64(10.0)
        exponent += 1

    rounded_value = sign * rounded_m * (np.float64(10.0) ** np.float64(exponent))
    ndigits = sig_figs - 1 - exponent

    return np.float64(rounded_value), int(ndigits)


def round_to_n_significant_figures(
    value: Union[float, np.floating], 
    sig_figs: int = 3
) -> np.float64:
    """
    Round a scalar to `sig_figs` significant figures using base-10 normalization.
    """
    v = np.asarray(value, dtype=np.float64)
    if v == 0.0:
        return np.float64(0.0)
    sign, mantissa, exponent = _normalize_base10(v)
    decimals = max(sig_figs - 1, 0)
    rounded_m = np.round(mantissa, decimals).astype(np.float64)
    if rounded_m >= np.float64(10.0):
        rounded_m = rounded_m / np.float64(10.0)
        exponent += 1
    return np.float64(sign * rounded_m * (np.float64(10.0) ** np.float64(exponent)))


def copy_precision(
    value: Union[float, np.floating], 
    precision: Union[float, np.floating]
) -> np.float64:
    """
    Round `value` to the same number of decimal places as `precision`.
    Example: precision = 14.8, value = 13.72342 -> 13.7
    This implementation finds the number of decimals in the decimal representation
    of `precision` and applies numpy.round with that decimals.
    """
    precision_decimal_places = abs(precision - int(precision))
    
    if precision_decimal_places > 0:
        num_decimal_places = len(str(precision).split('.')[1])
    else:
        num_decimal_places = 0
    
    return np.asarray(
        np.round(
            np.asarray(value, dtype=np.float64), num_decimal_places
        ), dtype=np.float64
    )


def cerv(df: pd.DataFrame, val_column_name: str, mode: str = "gost") -> pd.DataFrame:
    """
    Calculate Errors and Round Values (CERV)
    Round error column and corresponding measured values in-place in the DataFrame.
    The DataFrame must contain a column named f"Δ{val_column_name}".
    """
    error_column_name = f"Δ{val_column_name}"
    rounded = df[error_column_name].apply(lambda x: regop_val(x, mode=mode))
    rounded_error_values = rounded.apply(lambda t: t[0])
    precision_orders = rounded.apply(lambda t: int(t[1]))

    def _round_val(val, order):
        try:
            val_np = np.asarray(val, dtype=np.float64)
            order_int = int(order)
            return np.asarray(np.round(val_np, order_int), dtype=np.float64)
        except Exception:
            return val

    df[val_column_name] = df[val_column_name].combine(precision_orders, _round_val)
    df[error_column_name] = rounded_error_values.apply(lambda x: np.asarray(x, dtype=np.float64))
    return df


def remove_tail_from_magnitude(value: Union[float, np.floating]) -> np.float64:
    """
    Helper for pretty printing: round to 4 decimals for display.
    """
    return np.asarray(np.round(np.asarray(value, dtype=np.float64), 4), dtype=np.float64)


def get_pretty_expression(
    av: Union[float, np.floating], 
    d: Union[float, np.floating], 
    mode: str = "gost"
):
    """
    Produce a human-friendly string for value ± error scaled by magnitude.
    Returns formatted string, av, d, eps_fraction.
    """
    av_np = np.asarray(av, dtype=np.float64)
    d_np = np.asarray(d, dtype=np.float64)
    pretty_value = pd.DataFrame({"av": [av_np], "Δav": [d_np]})
    pretty_value = cerv(pretty_value, "av", mode=mode)
    eps_fraction = regop_val(
        pretty_value["Δav"].iat[0] / pretty_value["av"].iat[0], 
        mode=mode
    )[0]
    av_new = np.asarray(pretty_value["av"].iat[0], dtype=np.float64)
    d_new = np.asarray(pretty_value["Δav"].iat[0], dtype=np.float64)

    order_of_magnitude = int(np.floor(np.log10(np.abs(d_new))))
    factor = np.float64(10.0) ** np.float64(-order_of_magnitude)
    formatted = (
        f"({remove_tail_from_magnitude(av_new * factor)} ± "
        f"{remove_tail_from_magnitude(d_new * factor)}) 10^"
        f"{order_of_magnitude}, eps = {eps_fraction * 100}%"
    )
    return formatted, av_new, d_new, np.asarray(eps_fraction, dtype=np.float64)


def get_expr_and_average_and_total_error_and_eps(
    table: pd.DataFrame, 
    col: str, 
    mode: str = "gost", 
    confidence: float = 0.68
):
    """
    Compute mean, random and instrument errors, combine them according to rules
    and return pretty expression (delegates to get_pretty_expression).
    Random error is scaled by Student's t coefficient for the given confidence level.
    """
    values = table[col].to_numpy(dtype=np.float64)
    av = np.float64(np.mean(values))
    N = len(values)

    if N < 2:
        raise ValueError("Need at least 2 values to estimate random error")

    sigma = np.sqrt(np.sum((values - av) ** 2) / (N * (N - 1)))
    t_coeff = t.ppf((1 + confidence) / 2, df=N - 1)
    Sx = np.float64(t_coeff * sigma)

    Sx_pribor = np.sqrt(
        np.sum(np.asarray(table[f"Δ{col}"].to_numpy(dtype=np.float64)) ** 2) / N
    )

    print(f"Errors for {col} at {confidence*100:.1f}% CL:")
    print(f"\tRandom (t*σ): {Sx / av}")
    print(f"\tInstrument:  {Sx_pribor / av}")

    if Sx_pribor >= 3 * Sx:
        print("Using instrument error only")
        d = Sx_pribor
    elif Sx >= 3 * Sx_pribor:
        print("Using sample-spread (random) error only")
        d = Sx
    else:
        print("Combining both errors")
        d = np.float64(np.sqrt(Sx ** 2 + Sx_pribor ** 2))

    return get_pretty_expression(av, d, mode=mode)


def central_diff(f: np.ndarray, x: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Compute n-th order derivative using central differences (numpy.gradient repeatedly).
    """
    f_np = np.asarray(f, dtype=np.float64)
    x_np = np.asarray(x, dtype=np.float64)
    h = x_np[1] - x_np[0]
    for _ in range(n):
        f_np = np.gradient(f_np, h)
    return np.asarray(f_np, dtype=np.float64)


def compute_simpson_error(
    H: np.ndarray, 
    B_plus: np.ndarray, 
    B_minus: np.ndarray
) -> np.float64:
    """
    Estimate Simpson's rule error using an approximate fourth derivative of the integrand.
    """
    H_np = np.asarray(H, dtype=np.float64)
    B_plus_np = np.asarray(B_plus, dtype=np.float64)
    B_minus_np = np.asarray(B_minus, dtype=np.float64)

    B_diff = B_plus_np - B_minus_np
    f4 = central_diff(B_diff, H_np, n=4)
    max_f4 = np.max(np.abs(f4))
    N = len(H_np) - 1
    a, b = np.min(H_np), np.max(H_np)
    error = - (b - a) ** 5 / (2880 * (N ** 4)) * max_f4
    return np.asarray(np.abs(error), dtype=np.float64)


def compute_loop_area_with_error(
    H: np.ndarray, 
    B_plus: np.ndarray, 
    B_minus: np.ndarray
) -> Tuple[np.float64, np.float64]:
    """
    Compute the area between B_plus and B_minus using Simpson's rule and estimate its error.
    """
    H_np = np.asarray(H, dtype=np.float64)
    B_plus_np = np.asarray(B_plus, dtype=np.float64)
    B_minus_np = np.asarray(B_minus, dtype=np.float64)

    area = np.asarray(simpson(y=np.abs(B_plus_np - B_minus_np), x=H_np), dtype=np.float64)
    error_estimate = compute_simpson_error(H_np, B_plus_np, B_minus_np)
    return area, error_estimate
