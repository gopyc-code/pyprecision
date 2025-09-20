# test_pyprecision.py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pyprecision as pp


def test_normalize_base10_basic():
    sign, mantissa, exponent = pp._normalize_base10(1234.0)
    assert sign == np.float64(1.0)
    assert_allclose(mantissa * (10.0 ** exponent), np.float64(1234.0), rtol=0, atol=0)


def test_normalize_base10_small_and_negative():
    sign, mantissa, exponent = pp._normalize_base10(-0.00056)
    assert sign == np.float64(-1.0)
    assert 1.0 <= float(mantissa) < 10.0
    assert_allclose(sign * mantissa * (10.0 ** exponent), np.float64(-0.00056), rtol=1e-12)


@pytest.mark.parametrize(
    "x,mode,expected_val",
    [
        (1.01, "simple", 1.0),
        (1.49, "simple", 1.5),
        (1.8, "simple", 2.0),
        (6.9, "simple", 7.0),
    ],
)
def test_regop_val_simple_examples(x, mode, expected_val):
    val, nd = pp.regop_val(x, mode=mode)
    assert_allclose(val, expected_val, rtol=0, atol=1e-12)
    assert isinstance(nd, int)


def test_regop_val_gost_and_zero():
    v, nd = pp.regop_val(0.000184, mode="gost")
    assert_allclose(v, 0.00018, rtol=0, atol=1e-12)

    vz, ndz = pp.regop_val(0.0, mode="gost")
    assert vz == np.float64(0.0)
    assert ndz == 100


def test_regop_val_additional_cases():
    cases = [
        (14.8, "simple", 15.0),
        (13.72342, "gost", 14.0),
        (0.00456, "simple", 0.005),
        (0.000123, "gost", 0.00012),
        (0.0723, "gost", 0.07),
        (0.0323, "gost", 0.032),
        (100.0, "simple", 100.0),
    ]
    for val_in, mode, expected in cases:
        val, nd = pp.regop_val(val_in, mode=mode)
        assert_allclose(val, expected, rtol=0, atol=1e-12)
        assert isinstance(nd, int)


def test_round_to_n_significant_figures():
    test_cases = [
        (123.456, 3, 123.0),
        (0.004567, 2, 0.0046),
        (98765.4321, 4, 98770.0),
        (0.000123456, 2, 0.00012),
        (1000.0, 1, 1000.0),
        (0.0, 3, 0.0)
    ]
    
    for value, sig_figs, expected in test_cases:
        result = pp.round_to_n_significant_figures(value, sig_figs)
        assert_allclose(result, expected, rtol=1e-12)


def test_copy_precision_cases():
    test_cases = [
        (13.72342, 14.8, 13.7),
        (2.34567, 2.3, 2.3),
        (0.123456, 0.1, 0.1),
        (100.0, 1.0, 100.0),
        (0.0, 0.0, 0.0),
        (123.456, 123.4, 123.5),
        (1.23456789, 1.23, 1.23),
    ]
    for value, precision, expected in test_cases:
        res = pp.copy_precision(value, precision)
        assert_allclose(res, expected, rtol=1e-12)


def test_cerv_behavior_and_dtypes():
    df = pd.DataFrame({
        "value": [1.0, 3.5, 1.211, 1.16565],
        "Δvalue": [0.1, 2.0, 0.12, 0.05]
    })
    out = pp.cerv(df.copy(), "value", mode="simple")

    expected = np.array([1.0, 4.0, 1.21, 1.17], dtype=np.float64)
    assert_allclose(out["value"].to_numpy(dtype=np.float64), expected, rtol=0, atol=1e-12)


def test_remove_tail_from_magnitude():
    val = pp.remove_tail_from_magnitude(3.1415926535)
    assert_allclose(val, np.round(3.1415926535, 4), rtol=0)


if __name__ == "__main__":
    pytest.main(["-q", __file__])
