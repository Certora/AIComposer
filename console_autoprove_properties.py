"""Entry point for the known-properties auto-prove pipeline — console mode."""

if __name__ == "__main__":
    import sys
    from composer.cli.console_autoprove_properties import main
    sys.exit(main())
