from argparse import ArgumentParser

def bind_uid_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--uid",
        default=None,
        help="User id to list runs for. Defaults to value of $AUTOPROVER_USER_ID (which itself defaults to _anonymous).",
    )
