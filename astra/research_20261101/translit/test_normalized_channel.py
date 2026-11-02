import math
import unittest
from collections import defaultdict
from itertools import product
from normalized_channel import OPERATIONS, event_distribution, path_coefficients, score, support

PARAMS = {"operation_probabilities": {"delete": .4, "dup": .35, "swap": .25},
          "step_probabilities": {"0": .6, "1": .3, "2": .1}}


def direct_paths(clean, seq):
    """Independent positional enumerator; intentionally preserves duplicate outputs."""
    if not seq:
        return [(clean, 1.0)]
    n = len(clean)
    op = seq[0]
    if not n or (op == "swap" and n < 2):
        first = [(clean, 1.0)]
    elif op == "delete":
        first = [(clean[:i] + clean[i+1:], 1/n) for i in range(n)]
    elif op == "dup":
        first = [(clean[:i+1] + clean[i] + clean[i+1:], 1/n) for i in range(n)]
    else:
        first = [(clean[:i] + clean[i+1] + clean[i] + clean[i+2:], 1/(n-1)) for i in range(n-1)]
    return [(q, p1*p2) for middle, p1 in first for q, p2 in direct_paths(middle, seq[1:])]


class TestNormalizedChannel(unittest.TestCase):
    def test_total_mass(self):
        for clean in ("", "A", "AA", "AB", "ABA", "ABC", "가가", "가나다"):
            for op in OPERATIONS:
                self.assertAlmostEqual(sum(event_distribution(clean, op).values()), 1.0, places=12)
            self.assertAlmostEqual(math.fsum(score(clean, q, PARAMS) for q in support(clean)), 1.0, places=12)

    def test_direct_composition_coefficients(self):
        for clean in ("", "A", "AA", "AB", "ABA", "ABC"):
            expected = defaultdict(lambda: defaultdict(float))
            for k in range(3):
                for seq in product(OPERATIONS, repeat=k):
                    for query, mass in direct_paths(clean, seq):
                        expected[query]["/".join(seq)] += mass
            self.assertEqual(support(clean), set(expected))
            for query, coeff in expected.items():
                actual = path_coefficients(clean, query)
                self.assertEqual(set(actual), set(coeff))
                for key, value in coeff.items():
                    self.assertAlmostEqual(actual[key], value, places=12)

    def test_overlapping_events_and_identity(self):
        self.assertGreater(path_coefficients("ABC", "BCA")["swap/swap"], 0)
        self.assertAlmostEqual(path_coefficients("A", "AAA")["dup/dup"], 1.0)
        self.assertGreater(path_coefficients("A", "A")["dup/delete"], 0)
        self.assertAlmostEqual(path_coefficients("AA", "AA")["swap"], 1.0)
        self.assertAlmostEqual(score("", "", PARAMS), 1.0)
        self.assertEqual(path_coefficients("ABC", "ZZZ"), {})

    def test_order_independence(self):
        pairs = [("AB", "BA"), ("AA", "AAA"), ("", ""), ("ABC", "BCA")]
        first = {(a,b): (path_coefficients(a,b), score(a,b,PARAMS)) for a,b in pairs}
        second = {(a,b): (path_coefficients(a,b), score(a,b,PARAMS)) for a,b in reversed(pairs)}
        self.assertEqual(first, second)
        scratch = path_coefficients("AB", "BA")
        scratch.clear()
        self.assertEqual(path_coefficients("AB", "BA"), first[("AB", "BA")][0])


if __name__ == "__main__":
    unittest.main()
