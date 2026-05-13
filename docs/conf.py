"""Sphinx configuration for the project documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

project = "pallatom"
author = "tmatteso"
release = "0.1.0"
copyright = "2024, tmatteso"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
    "sphinxcontrib.autodoc_pydantic",
]

# ---------------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------------
autoapi_dirs = ["../pallatom"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
# Exclude test files from API docs
autoapi_ignore = ["*/tests/*"]
# Include both class-level and __init__ docstrings
autoapi_python_class_content = "both"
# Group by type (class, function, attribute) rather than alphabetically
autoapi_member_order = "groupwise"
# Retain generated .rst files so they can be inspected / customised
autoapi_keep_files = True

# ---------------------------------------------------------------------------
# autodoc-pydantic  (renders Pydantic models with field tables & validators)
# ---------------------------------------------------------------------------
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_settings_show_json = True

# ---------------------------------------------------------------------------
# Napoleon  (Google / NumPy-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# ---------------------------------------------------------------------------
# Intersphinx cross-references
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "pallatom"
html_static_path = ["_static"]
