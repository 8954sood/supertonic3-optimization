import json
import collections
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("report")
a = p.parse_args()
r = json.loads(Path(a.report).read_text())
for filename in r.get("profiles", []):
    ops = collections.Counter()
    for e in json.loads(Path(filename).read_text()):
        if e.get("cat") == "Node":
            ops[e.get("args", {}).get("op_name", "?")] += e.get("dur", 0)
    total = sum(ops.values())
    print(filename)
    for name, cost in ops.most_common(10):
        print(f"  {name:25s} {cost / 1e6:.3f}s {100 * cost / max(total, 1):.1f}%")
