import os
import sys

env = os.environ.get("CERTORA")
if env is None:
    raise RuntimeError("CERTORA environment variable not set")
if env not in sys.path:
    sys.path.append(env)
