import unittest
from runtime import Engine
import numpy as np


class OptimizedChecks(unittest.TestCase):
    def test_fp32_cached_across_voice_steps_and_length(self):
        for voice in ("M1", "F1"):
            baseline = Engine(voice=voice)
            optimized = Engine(model="models/fast-fp32-cached", voice=voice)
            for lang, text, steps in [
                ("ko", "아침 공기가 맑습니다.", 8),
                ("en", "Please check the window before leaving the house.", 12),
            ]:
                with self.subTest(voice=voice, lang=lang, steps=steps):
                    a, _ = baseline.synthesize(text, lang, steps, seed=7)
                    b, _ = optimized.synthesize(text, lang, steps, seed=7)
                    self.assertEqual(len(a), len(b))
                    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                    print(voice, lang, steps, "rmse", rmse, flush=True)
                    self.assertLess(rmse, 0.0001)
            del baseline, optimized


if __name__ == "__main__":
    unittest.main()
