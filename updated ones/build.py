"""Build web/index.html (one self-contained file) from the Python project results.
Run from anywhere:  python web/build.py      (needs: Python project already run with `python run_all.py`, and Node.js)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, GEN = os.path.join(HERE, "src"), os.path.join(HERE, "src", "generated")
subprocess.check_call([sys.executable, os.path.join(SRC, "make_sample.py")])
subprocess.check_call(["node", os.path.join(SRC, "make_model.js")])
subprocess.check_call([sys.executable, os.path.join(SRC, "make_static.py")])
read = lambda p: open(p, encoding="utf-8").read()
html = (read(os.path.join(SRC, "template.html")).replace("/*ENGINE*/", read(os.path.join(SRC, "engine.js")))
        .replace("__SAMPLE__", read(os.path.join(GEN, "sample.csv")).strip())
        .replace("__STATIC__", read(os.path.join(GEN, "static.json"))).replace("__MODEL__", read(os.path.join(GEN, "model.json"))))
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(html)
print("web/index.html built:", round(len(html) / 1e6, 2), "MB")
