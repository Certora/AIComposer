"""Entry point for the auto-prove multi-agent pipeline TUI."""

import sys

if __name__ == "__main__":
    from composer.cli.tui_autoprove import main
    sys.exit(main())
