import pathlib
import sys

verisafe_dir = str(pathlib.Path(__file__).parent.parent.absolute())

if verisafe_dir not in sys.path:
    sys.path.append(verisafe_dir)

import composer.certora as _

