"""Entry point for the known-properties auto-prove pipeline TUI."""

if __name__ == "__main__":
    import sys
    from composer.cli.tui_autoprove_properties import main
    sys.exit(main())
