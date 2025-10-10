import pathlib
import sys

verisafe_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())

if verisafe_dir not in sys.path:
    sys.path.append(verisafe_dir)

import verisafe.certora as _

