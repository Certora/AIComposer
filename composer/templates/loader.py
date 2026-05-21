from typing import Any
from jinja2 import Environment, FileSystemLoader
import pathlib

script_dir = pathlib.Path(__file__).parent
env = Environment(loader=FileSystemLoader(script_dir))

def _list_templates_in(subdir: str) -> list[str]:
    target = script_dir / subdir
    if not target.is_dir():
        return []
    return sorted(
        f"{subdir.rstrip('/')}/{p.name}"
        for p in target.iterdir()
        if p.is_file() and p.suffix == ".j2"
    )

env.globals["list_templates_in"] = _list_templates_in

def load_jinja_template(template_name: str, **kwargs: Any) -> str:
    """Load and render a Jinja template from the script directory"""
    template = env.get_template(template_name)
    return template.render(**kwargs)
