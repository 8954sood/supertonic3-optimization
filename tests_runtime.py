"""Checks against the upstream engine, not a duplicate of implementation logic."""

import unittest
from runtime import Engine
import numpy as np
from supertonic.core import Supertonic


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_upstream_equivalence_eight_steps(self):
        e = self.engine
        official = Supertonic(
            e.cfg,
            e.processor,
            *[
                e.sessions[k]
                for k in (
                    "duration_predictor",
                    "text_encoder",
                    "vector_estimator",
                    "vocoder",
                )
            ],
        )
        for lang, text in [
            ("ko", "안녕하세요. 음성 합성 속도를 확인합니다."),
            ("en", "A quiet morning by the window."),
        ]:
            with self.subTest(lang=lang):
                np.random.seed(42)
                expected, duration = official(
                    [text], e.style, total_step=8, speed=1.05, lang=lang
                )
                actual, _ = e.synthesize(text, lang, 8, seed=42)
                expected = expected.reshape(-1)[
                    : int(float(duration[0]) * e.sample_rate)
                ]
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_requests(self):
        for kwargs in (
            {"text": ""},
            {"text": "Hello", "steps": 7},
            {"text": "Hello", "speed": 0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.engine.synthesize(**kwargs)


if __name__ == "__main__":
    unittest.main()
