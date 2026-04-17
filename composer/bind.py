try:
    import graphcore.graph
except ImportError:
    import pathlib
    composer_project_root = pathlib.Path(__file__).parent.parent
    if (graphcore_dir := (composer_project_root / "graphcore")).exists() and graphcore_dir.is_dir():
        import importlib
        import sys
        if "graphcore" in sys.modules:
            del sys.modules["graphcore"]
        sys.path.insert(0, str(graphcore_dir))
        importlib.invalidate_caches()

import logging

logging.getLogger("huggingface_hub.utils._http").addFilter(
    lambda r: "You are sending unauthenticated requests to the HF Hub." not in r.getMessage()
)
