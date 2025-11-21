import logging
import sys
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

import click

from mnemolet_core.config import (
    QDRANT_URL,
)
from mnemolet_core.core.utils.qdrant import QdrantManager

logger = logging.getLogger(__name__)

try:
    __version__ = version("mnemolet_core")
except PackageNotFoundError:
    __version__ = "0.1.0"


def requires_qdrant(f):
    """
    Decorator to check Qdrant before running a command.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        qm = QdrantManager(QDRANT_URL)
        if not qm.check_qdrant_status():
            sys.exit(1)
        return f(*args, **kwargs)

    return wrapper


@click.group()
@click.option(
    "-v", "--verbose", count=True, help="Increase output verbosity (use -v or -vv)"
)
@click.version_option(
    version=__version__,
    prog_name="MnemoLet",
)
@click.pass_context
def cli(ctx, verbose):
    """
    CLI for mnemolet.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # map -v to INFO, --vv to DEBUG
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:  # -vv or more
        level = logging.DEBUG

    # reset existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    logger.debug(f"Logger init with level={level}")


def register_commands():
    from mnemolet_core.cli.commands.answer import answer
    from mnemolet_core.cli.commands.config import init_config
    from mnemolet_core.cli.commands.ingest import ingest
    from mnemolet_core.cli.commands.list import list_collections_cli
    from mnemolet_core.cli.commands.remove import remove
    from mnemolet_core.cli.commands.search import search
    from mnemolet_core.cli.commands.stats import stats

    cli.add_command(init_config)
    cli.add_command(ingest)
    cli.add_command(search)
    cli.add_command(answer)
    cli.add_command(stats)
    cli.add_command(list_collections_cli)
    cli.add_command(remove)


register_commands()

if __name__ == "__main__":
    cli()
