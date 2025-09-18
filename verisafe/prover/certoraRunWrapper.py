#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import pickle
import os

"""
This is a wrapper script which sandboxes an invocation of run_certora.py
while still allowing access to the structured return type.

The structured data is returned via the first argument, which is a temp file
into which the serialized result is written (either None or a CertoraRunResult).

If certora run throws an exception, it is caught, and the serialized representation of the
exception is written to the same file.

All other arguments past the first are passed through to `run_certora`.
"""

os.putenv("DONT_USE_VERIFICATION_RESULTS_FOR_EXITCODE", "1")

certora_path = os.environ.get("CERTORA")
if certora_path is None:
    sys.exit(1)

sys.path.append(certora_path)

from certoraRun import run_certora

output = sys.argv[1]

with open(output, 'wb') as out:
    try:
        r = run_certora(
            args=sys.argv[2:]
        )
        pickle.dump(r, out)
    except Exception as e:
        pickle.dump(e, out)

sys.exit(0)
