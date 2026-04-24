"""Entry point for the NatSpec multi-agent pipeline TUI."""

import sys

if __name__ == "__main__":
    from composer.cli.tui_pipeline import main
    sys.exit(main())
