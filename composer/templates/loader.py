from typing import Any
from jinja2 import Environment, FileSystemLoader
import pathlib

script_dir = pathlib.Path(__file__).parent


def _autoescape(template_name: str | None) -> bool:
    # HTML templates (``*.html.j2``) must autoescape interpolated values; prompt templates
    # (plain ``.j2``) stay verbatim — escaping would corrupt their contents.
    return template_name is not None and template_name.endswith(".html.j2")


env = Environment(loader=FileSystemLoader(script_dir), autoescape=_autoescape)

def load_jinja_template(template_name: str, **kwargs: Any) -> str:
    """Load and render a Jinja template from the script directory"""
    template = env.get_template(template_name)
    return template.render(**kwargs)
