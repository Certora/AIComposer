"""
Advisory CVL typecheck tool for the CVL author agent.

The author agent writes its current working spec into state via
``put_cvl`` / ``put_cvl_raw``; this tool reads that out, materializes the
project view via the caller-supplied assembler, and invokes
``certoraTypeCheck.py`` against the combined tree. Returns success/failure
as human-readable strings — the tool is advisory, used to catch type errors
before the author commits a result.

This lived in ``merge.py`` until the merge agent was removed (the natspec
deliverable is now ``contract -> list of specs`` with no post-agent merge
step). The typecheck tool is the only piece worth keeping from that module.
"""

import asyncio
import pathlib
import sys
from typing import Callable, Awaitable



from composer.spec.system_model import SolidityIdentifier
from composer.spec.natspec.registry import FileRegistry
from composer.spec.natspec.task_description import Assembler, ConfigurationBuilder
from composer.spec.util import temp_certora_file


async def typecheck_spec(
    files: list[str],
    *,
    spec: str,
    primary_contract: SolidityIdentifier,
    assembler: Assembler,
    config_builder: ConfigurationBuilder,
) -> str | None:
    """Run certoraTypeCheck.py on ``spec`` against the assembled project.

    The assembler lays out interfaces / stubs / existing source in a tmpdir,
    the spec is written alongside, and the merged Certora conf (``files``,
    ``verify``, plus whatever the caller seeded in ``config_builder``) is
    written too. Returns ``None`` on success or a human-readable error string
    on failure.
    """
    async with assembler.project_directory() as tmpdir:
        with (
            temp_certora_file(
                content=spec,
                root=str(tmpdir),
                ext="spec",
            ) as spec_file,
            (
                config_builder
                .with_files(files)
                .with_verify(main_contract=primary_contract, spec_file=spec_file)
                .build_to(tmpdir)
            ) as config_file,
        ):
            entry = (pathlib.Path(__file__).parent.parent / "certoraTypeCheck.py").absolute()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(entry), str(config_file),
                cwd=str(tmpdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            if proc.returncode == 0:
                return None
            return (
                f"stdout:\n{stdout_b.decode()}\n\n"
                f"stderr:\n{stderr_b.decode()}"
            )

type TypeChecker = Callable[[str], Awaitable[str | None]]

def make_typechecker(
    files: FileRegistry,
    assembler: Assembler,
    config_builder: ConfigurationBuilder,
    primary_contract: SolidityIdentifier
) -> TypeChecker:
    async def to_return(
        spec: str
    ) -> str | None:
        return await typecheck_spec(
            files=await files.read_all(primary_contract),
            primary_contract=primary_contract,
            assembler=assembler,
            spec=spec,
            config_builder=config_builder
        )
    return to_return
