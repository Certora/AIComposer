"""Entry point for the auto-prove pipeline — console (no TUI) mode."""

if __name__ == "__main__":
    import sys
    from composer.cli.console_autoprove import main
    sys.exit(main())
