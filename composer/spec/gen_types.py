
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping, Any, Protocol

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Canonical certora/ layout
#
# Every persisted path variable in the spec pipeline is stored **relative to the
# project root** (e.g. ``certora/specs/invariants.spec``). Conversions to other
# bases happen only at the edges: ``dump_final_conf`` emits the verify entry
# verbatim (the prover reads it relative to the project root), and CVL ``import``
# statements are derived with :func:`import_statement_for` (the prover reads
# those relative to the importing spec's own directory).
# ---------------------------------------------------------------------------
CERTORA_DIR = Path("certora")
#: Generated specs (the "importers") are written here.
SPECS_DIR = CERTORA_DIR / "specs"
#: AutoSetup / custom summaries live here.
SUMMARIES_DIR = SPECS_DIR / "summaries"


def under_project(project_root: "str | Path", rel: "str | Path") -> Path:
    """Resolve a project-root-relative path (e.g. :data:`CERTORA_DIR`,
    :data:`SPECS_DIR`, or a canonical spec path) to an absolute path under
    *project_root*. The single place the canonical layout meets a concrete root.
    """
    return Path(project_root) / rel


def certora_relative_to_project(p: str) -> Path:
    """Express *p* -- a path relative to the ``certora/`` directory -- relative to
    the project root, by prefixing ``certora/``.

    This is the form the external AutoSetup tool reports its summaries path in
    (e.g. ``specs/summaries/X.spec`` -> ``certora/specs/summaries/X.spec``): see
    ``ai_autoprover/autosetup/handler.py``, which builds it via
    ``resolved_summary_path.relative_to(project_root / "certora")``. The assert
    guards that invariant so a future format change fails loudly here rather than
    silently producing a ``certora/certora/...`` path downstream.
    """
    pp = PurePosixPath(p)
    assert pp.parts[:len(CERTORA_DIR.parts)] != CERTORA_DIR.parts, (
        f"expected a certora/-relative path, got project-root-relative {p!r}"
    )
    return CERTORA_DIR / p


def import_statement_for(resource_path: Path, importer_dir: Path) -> str:
    """CVL import path to *resource_path* as seen from a spec located in
    *importer_dir* (both project-root-relative).

    The prover resolves a CVL ``import`` relative to the importing spec's own
    directory, so this is just *resource_path* expressed relative to *importer_dir*
    -- including ``..`` segments if the resource lives outside that directory's
    subtree.
    """
    return PurePosixPath(resource_path).relative_to(importer_dir, walk_up=True).as_posix()



# ---------------------------------------------------------------------------
# GenerationEnv — unified configuration for CVL generation
# ---------------------------------------------------------------------------

class CVLResource(BaseModel):
    path: Path = Field(description="path to the resource file, relative to the project root (e.g. `certora/specs/invariants.spec`)")
    required: bool = Field(description="whether this resource *must* be used in the verification process")
    description: str = Field(description="A description of this resource")
    sort: Literal["import"]

class TypedTemplate[T: Mapping[str, Any]]:
    def __init__(self, name: str):
        self._wrapped = name

    def __str__(self) -> str:
        return self._wrapped

    def bind(self, params: T) -> "TemplateInstantiation":
        return TemplateInstantiation.create(self, params)

class TemplateRenderer[T](Protocol):
    def __call__(self, template: str, **kwargs) -> T:
        ...

@dataclass
class TemplateInstantiation:
    template: TypedTemplate
    args: dict

    @staticmethod
    def create[T: Mapping[str, Any]](
        templ: TypedTemplate[T],
        args: T
    ) -> "TemplateInstantiation":
        assert isinstance(args, dict)
        return TemplateInstantiation(templ, args)

    def render_to[T](
        self,
        cb: TemplateRenderer[T]
    ) -> T:
        return cb(
            str(self.template),
            **self.args
        )

    def depends[X: Mapping[str, Any]](self, other: type[X]) -> "InjectedTemplate[X]":
        return InjectedTemplate(self)

@dataclass
class InjectedTemplate[X: Mapping[str, Any]]:
    wrapped: TemplateInstantiation

    def inject(self, injected: X) -> TemplateInstantiation:
        return TemplateInstantiation(
            TypedTemplate(str(self.wrapped.template)),
            {
                **self.wrapped.args,
                **injected
            }
        )
