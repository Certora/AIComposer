"""``ap-trail`` — autoprover run trail utility.

Subcommands:

* ``ls``     — list recent runs.
* ``view``   — drill-down TUI for one run's thread forest (live DB or
                ``--from-export`` replay).
* ``export`` — dump a run + per-thread timelines to a gzipped JSON file
                consumable by ``ap-trail view --from-export``.
* ``data``   — show the ``run_data`` metadata dicts recorded for a run
                (``--json`` for pipeable output).
"""

import argparse
import sys

from composer.cli.diagnostics import ap_trail_data, ap_trail_export, ap_trail_ls, ap_trail_view


def main() -> int:
    parser = argparse.ArgumentParser(prog="ap-trail", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="List recent runs.")
    ap_trail_ls.add_arguments(p_ls)

    p_view = sub.add_parser("view", help="Drill-down explorer for one run.")
    ap_trail_view.add_arguments(p_view)

    p_export = sub.add_parser("export", help="Export a run to a gzipped JSON file.")
    ap_trail_export.add_arguments(p_export)

    p_data = sub.add_parser("data", help="Show a run's recorded run_data metadata.")
    ap_trail_data.add_arguments(p_data)

    args = parser.parse_args()
    match args.cmd:
        case "ls":
            return ap_trail_ls.main(args)
        case "view":
            return ap_trail_view.main(args)
        case "export":
            return ap_trail_export.main(args)
        case "data":
            return ap_trail_data.main(args)
        case _:
            parser.print_help()
            return 2


if __name__ == "__main__":
    sys.exit(main())
