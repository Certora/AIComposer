import pathlib
import sys

composer_dir = str(pathlib.Path(__file__).parent.parent.absolute())

if composer_dir not in sys.path:
    sys.path.append(composer_dir)

import composer.certora as _

