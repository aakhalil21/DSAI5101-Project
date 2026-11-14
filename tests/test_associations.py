"""Synthetic tests for src/associations.py.

Run from the repository root with:  py -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from associations import (  # noqa: E402
    correlation_ratio,
    cramers_v_bc,
    group_bootstrap,
    spearman_assoc,
    sparse_groups,
)

KEYS = {"estimate", "n_used", "missing_pair_fraction"}


class SpearmanTests(unittest.TestCase):
    def test_perfect_monotonic_and_sign(self):
        x = pd.Series([1, 2, 3, 4, 5])
        self.assertAlmostEqual(spearman_assoc(x, x ** 3)["estimate"], 1.0)
        self.assertAlmostEqual(spearman_assoc(x, -x)["estimate"], -1.0)

    def test_known_value_with_ties(self):
        # Average ranks: x -> 1, 2.5, 2.5, 4; y -> 1, 2, 3, 4; Pearson of ranks = 0.9486833
        result = spearman_assoc([1, 2, 2, 3], [10, 20, 30, 40])
        self.assertAlmostEqual(result["estimate"], 3 / math.sqrt(10), places=7)

    def test_observed_pairs_only(self):
        x = pd.Series([1, 2, np.nan, 4, 5, 6])
        y = pd.Series([2, 4, 6, np.nan, 10, 12])
        result = spearman_assoc(x, y)
        self.assertEqual(KEYS, KEYS & set(result))
        self.assertEqual(result["n_used"], 4)
        self.assertAlmostEqual(result["missing_pair_fraction"], 2 / 6)
        self.assertAlmostEqual(result["estimate"], 1.0)

    def test_ordinal_map_sets_direction(self):
        health = pd.Series(["Poor", "Fair", "Good", "Very good", "Excellent"])
        days = pd.Series([30, 20, 10, 5, 0])
        up = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}
        down = {k: -v for k, v in up.items()}
        self.assertAlmostEqual(spearman_assoc(health, days, ordinal_map=up)["estimate"], -1.0)
        self.assertAlmostEqual(spearman_assoc(health, days, ordinal_map=down)["estimate"], 1.0)

    def test_non_numeric_without_map_or_unmapped_category_raises(self):
        with self.assertRaises(ValueError):
            spearman_assoc(["a", "b", "c"], [1, 2, 3])
        with self.assertRaises(ValueError):
            spearman_assoc(["a", "b", "z"], [1, 2, 3], ordinal_map={"a": 1, "b": 2})

    def test_constant_or_too_few_pairs_is_nan(self):
        self.assertTrue(math.isnan(spearman_assoc([1, 1, 1, 1], [1, 2, 3, 4])["estimate"]))
        self.assertTrue(math.isnan(spearman_assoc([1, 2], [3, 4])["estimate"]))
        self.assertTrue(math.isnan(spearman_assoc([np.nan] * 3, [1, 2, 3])["estimate"]))


class CramersVTests(unittest.TestCase):
    def test_variable_with_itself_is_one(self):
        x = pd.Series(list("abc") * 20)
        result = cramers_v_bc(x, x)
        self.assertAlmostEqual(result["estimate"], 1.0, places=10)
        self.assertEqual(result["table"].shape, (3, 3))

    def test_independent_table_is_zero(self):
        # Exactly proportional counts give chi-square 0, so the corrected V is 0.
        x = pd.Series(["a"] * 20 + ["b"] * 20)
        y = pd.Series((["u"] * 10 + ["v"] * 10) * 2)
        self.assertEqual(cramers_v_bc(x, y)["estimate"], 0.0)

    def test_shuffled_copy_near_zero(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.choice(list("abcd"), size=5000))
        shuffled = pd.Series(rng.permutation(x.to_numpy()))
        self.assertLess(cramers_v_bc(x, shuffled)["estimate"], 0.05)

    def test_bias_correction_formula(self):
        # 2x2 table [[30, 10], [10, 30]]: chi2 = 20, n = 80
        x = pd.Series(["a"] * 40 + ["b"] * 40)
        y = pd.Series(["u"] * 30 + ["v"] * 10 + ["u"] * 10 + ["v"] * 30)
        phi2_corr = 20 / 80 - 1 / 79
        expected = math.sqrt(phi2_corr / (2 - 1 / 79 - 1))
        self.assertAlmostEqual(cramers_v_bc(x, y)["estimate"], expected, places=10)

    def test_missing_pairs_and_constant(self):
        x = pd.Series(["a", "b", None, "a"])
        y = pd.Series(["u", "v", "u", None])
        result = cramers_v_bc(x, y)
        self.assertEqual(result["n_used"], 2)
        self.assertAlmostEqual(result["missing_pair_fraction"], 0.5)
        self.assertTrue(math.isnan(cramers_v_bc(["a"] * 10, list("uvuvuvuvuv"))["estimate"]))


class CorrelationRatioTests(unittest.TestCase):
    def test_known_value(self):
        # Group means 2 and 6, grand mean 4: between SS = 6*4 = 24; total SS = 24 + 2*2 = 28
        cats = ["a", "a", "a", "b", "b", "b"]
        values = [1, 2, 3, 5, 6, 7]
        self.assertAlmostEqual(correlation_ratio(cats, values)["estimate"], math.sqrt(24 / 28))

    def test_perfect_and_none(self):
        self.assertAlmostEqual(correlation_ratio(list("aabb"), [1, 1, 5, 5])["estimate"], 1.0)
        self.assertAlmostEqual(correlation_ratio(list("abab"), [1, 1, 5, 5])["estimate"], 0.0)

    def test_constant_is_nan(self):
        self.assertTrue(math.isnan(correlation_ratio(list("abab"), [3, 3, 3, 3])["estimate"]))
        self.assertTrue(math.isnan(correlation_ratio(list("aaaa"), [1, 2, 3, 4])["estimate"]))

    def test_missing_pairs(self):
        result = correlation_ratio(["a", None, "b", "b"], [1.0, 2.0, np.nan, 4.0])
        self.assertEqual(result["n_used"], 2)
        self.assertAlmostEqual(result["missing_pair_fraction"], 0.5)


class SparseGroupsTests(unittest.TestCase):
    def test_flags_below_threshold(self):
        x = pd.Series(["a"] * 30 + ["b"] * 29 + ["c"] * 2 + [None] * 5)
        out = sparse_groups(x).set_index("category")
        self.assertEqual(out.loc["a", "n"], 30)
        self.assertFalse(out.loc["a", "sparse"])
        self.assertTrue(out.loc["b", "sparse"])
        self.assertTrue(out.loc["c", "sparse"])
        self.assertNotIn("None", out.index)
        self.assertTrue(sparse_groups(x, min_n=2).set_index("category").loc["b", "sparse"] == False)  # noqa: E712


class GroupBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(size=n)
        self.df = pd.DataFrame({
            "x": x,
            "y": x + rng.normal(scale=0.5, size=n),
            "duplicate_group": np.arange(n) // 2,  # pairs of records share a group
        })

    def test_reproducible_and_brackets_estimate(self):
        func = lambda d: spearman_assoc(d["x"], d["y"])  # noqa: E731
        first = group_bootstrap(self.df, func, n_boot=100, seed=42)
        second = group_bootstrap(self.df, func, n_boot=100, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["n_groups"], 200)
        self.assertEqual(first["n_boot_valid"], 100)
        self.assertLess(first["ci_lower"], first["estimate"])
        self.assertGreater(first["ci_upper"], first["estimate"])

    def test_whole_groups_are_resampled(self):
        sizes = []

        def record(d):
            # Every group appears with both of its records, so counts are even.
            sizes.append(d.groupby("duplicate_group").size().mod(2).sum())
            return 0.0

        group_bootstrap(self.df, record, n_boot=20, seed=3)
        self.assertEqual(sum(sizes), 0)

    def test_nan_resamples_are_skipped(self):
        out = group_bootstrap(self.df, lambda d: np.nan, n_boot=5)
        self.assertEqual(out["n_boot_valid"], 0)
        self.assertTrue(math.isnan(out["ci_lower"]))


if __name__ == "__main__":
    unittest.main()
