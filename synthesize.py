import argparse
from runtime import Engine
import json
from pathlib import Path
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("text")
p.add_argument("--lang", default="ko")
p.add_argument("--voice", default="M1")
p.add_argument("--steps", type=int, default=8)
p.add_argument("--speed", type=float, default=1.05, help="Speaking rate: 0.7 to 2.0; 1.0 is normal")
p.add_argument("--variant")
p.add_argument("--model", default="models/supertonic-3")
p.add_argument("--output", default="samples/output.wav")
a = p.parse_args()
if not 0.7 <= a.speed <= 2.0:
    p.error("--speed must be between 0.7 and 2.0")
e = Engine(model=a.model, variant=a.variant, voice=a.voice)
w, m = e.synthesize(a.text, a.lang, a.steps, speed=a.speed)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
sf.write(a.output, w, e.sample_rate)
print(json.dumps(m, ensure_ascii=False, indent=2))
print(a.output)
