import sys,argparse
sys.path.insert(0,"/opt/forge")
import forge
forge.TRAIN_PY="/data/train_canonical.py"
forge.cmd_build(argparse.Namespace(name="samantha",from_step="generate",only_step=None,overwrite=False))
