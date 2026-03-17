from typing import Any
from jinja2 import Environment, FileSystemLoader
import pathlib

from composer.input.config import config

script_dir = pathlib.Path(__file__).parent
env = Environment(loader=FileSystemLoader(script_dir))

def load_jinja_template(template_name: str, **kwargs: Any) -> str:
    """Load and render a Jinja template from the script directory"""
    kwargs.update({"platform": config.platform})
    template = env.get_template(template_name)
    return template.render(**kwargs)
