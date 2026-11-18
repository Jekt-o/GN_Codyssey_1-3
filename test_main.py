"""표준 라이브러리만 사용하는 Mini NPU 회귀 테스트."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import main


class MiniNPUTests(unittest.TestCase):
    def setUp(self):
        self.patterns = main.generate_patterns(3)
        self.cross = self.patterns["Cross"]
        self.x = self.patterns["X"]
        self.bank, self.errors, self.global_error = main.load_filter_bank(
            {"size_3": {"cross": self.cross, "x": self.x}}
        )

    def analyze(self, case_id, raw_case):
        return main.analyze_case(
            case_id, raw_case, self.bank, self.errors, self.global_error
        )

    def test_mac_matches_assignment_examples(self):
        self.assertEqual(main.mac(self.cross, self.cross), 5.0)
        self.assertEqual(main.mac(self.cross, self.x), 1.0)
        self.assertEqual(main.mac(self.x, self.x), 5.0)
        self.assertAlmostEqual(main.mac([[0.1, 0.2], [0.3, -0.4]],
                                        [[1, 2], [3, 4]]), -0.2)

    def test_label_normalization_and_unknown_labels(self):
        for source, expected in (("+", "Cross"), ("cross", "Cross"),
                                 ("Cross", "Cross"), ("x", "X"), ("X", "X")):
            with self.subTest(source=source):
                self.assertEqual(main.normalize_label(source), expected)
        for source in ("triangle", "UNDECIDED", None, 1, ["x"]):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    main.normalize_label(source)

    def test_epsilon_policy_is_strict_at_boundary(self):
        self.assertEqual(main.decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(main.decide(main.EPSILON / 2, 0.0), "UNDECIDED")
        self.assertEqual(main.decide(main.EPSILON, 0.0), "Cross")
        self.assertEqual(main.decide(0.0, main.EPSILON), "X")
        self.assertEqual(main.decide(1.0, 5.0, "A", "B"), "B")

    def test_matrix_validation_rejects_shape_and_nonfinite_values(self):
        invalid_matrices = (
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5], [6, 7, 8]],
            [[1, 2, 3], [4, "5", 6], [7, 8, 9]],
            [[1, 2, 3], [4, True, 6], [7, 8, 9]],
            [[1, 2, 3], [4, float("nan"), 6], [7, 8, 9]],
            [[1, 2, 3], [4, float("inf"), 6], [7, 8, 9]],
            None,
        )
        for value in invalid_matrices:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    main.validate_matrix(value, 3)

    def test_mac_overflow_is_reported(self):
        with self.assertRaises(ValueError):
            main.mac([[1e308]], [[1e308]])

    def test_normalized_expected_labels_pass(self):
        for index, pattern, expected, label in (
            (1, self.cross, "+", "Cross"),
            (2, self.x, "x", "X"),
        ):
            result = self.analyze("size_3_{}".format(index),
                                  {"input": pattern, "expected": expected})
            self.assertTrue(result.passed, result.reason)
            self.assertEqual(result.predicted, label)
            self.assertEqual(result.expected, label)

    def test_tie_and_wrong_expected_are_failures(self):
        tie = self.analyze("size_3_1", {"input": [[0] * 3 for _ in range(3)],
                                        "expected": "x"})
        self.assertFalse(tie.passed)
        self.assertEqual(tie.predicted, "UNDECIDED")
        self.assertTrue(tie.reason)
        mismatch = self.analyze("size_3_2", {"input": self.cross, "expected": "x"})
        self.assertFalse(mismatch.passed)
        self.assertEqual(mismatch.predicted, "Cross")
        self.assertTrue(mismatch.reason)

    def test_bad_case_does_not_prevent_later_valid_case(self):
        bad_cases = (
            ("bad_key", {"input": self.cross, "expected": "+"}),
            ("size_5_1", {"input": self.cross, "expected": "+"}),
            ("size_3_1", {"input": [[0]], "expected": "+"}),
            ("size_3_2", {"input": self.cross, "expected": "?"}),
            ("size_3_3", {"expected": "+"}),
            ("size_3_4", None),
        )
        for case_id, raw_case in bad_cases:
            with self.subTest(case_id=case_id):
                result = self.analyze(case_id, raw_case)
                self.assertFalse(result.passed)
                self.assertTrue(result.reason)
        self.assertTrue(self.analyze("size_3_5", {"input": self.x,
                                                  "expected": "x"}).passed)

    def test_bad_filter_isolated_from_other_sizes(self):
        bank, errors, global_error = main.load_filter_bank({
            "size_3": {"cross": self.cross, "x": self.x},
            "size_5": {"cross": [[1]], "x": [[1]]},
        })
        bad = main.analyze_case("size_5_1", {"input": [[0] * 5 for _ in range(5)],
                                             "expected": "+"},
                                bank, errors, global_error)
        good = main.analyze_case("size_3_1", {"input": self.x, "expected": "x"},
                                 bank, errors, global_error)
        self.assertFalse(bad.passed)
        self.assertTrue(good.passed, good.reason)

    def run_cli(self, args, user_input=""):
        return subprocess.run(
            [sys.executable, str(Path(main.__file__).resolve())] + args,
            input=user_input, text=True, capture_output=True, timeout=10,
        )

    def test_cli_json_continues_and_summarizes_failures(self):
        document = {
            "filters": {"size_3": {"cross": self.cross, "x": self.x}},
            "patterns": {
                "size_3_1": {"input": [[1]], "expected": "+"},
                "size_3_2": {"input": self.x, "expected": "x"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            result = self.run_cli(["--mode", "2", "--data", str(path),
                                   "--repeats", "10"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("size_3_1", result.stdout)
        self.assertIn("size_3_2", result.stdout)
        self.assertIn("PASS", result.stdout)
        self.assertIn("FAIL", result.stdout)
        for label, count in (("총 테스트", 2), ("통과", 1), ("실패", 1)):
            self.assertRegex(result.stdout, label + r"\s*:\s*" + str(count))

    def test_cli_retries_bad_row_and_handles_end_of_input(self):
        rows = ["not a number", "0 1", "0 1 0", "1 1 1", "0 1 0",
                "1 0 1", "0 1 0", "1 0 1", "1 0 1", "0 1 0", "1 0 1"]
        result = self.run_cli(["--mode", "1", "--repeats", "10"],
                               "\n".join(rows) + "\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"판정\s*:\s*B")
        self.assertIn("오류", result.stdout)
        incomplete = self.run_cli(["--mode", "1", "--repeats", "10"], "0 1 0\n")
        self.assertNotIn("Traceback", incomplete.stderr)
        self.assertNotIn("Traceback", incomplete.stdout)

    def test_cli_rejects_fewer_than_ten_repetitions(self):
        result = self.run_cli(["--mode", "2", "--repeats", "9"])
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
