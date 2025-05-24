# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'city2graph'
copyright = '2025, Yuta Sato & city2graph developers'
author = 'Yuta Sato'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinxext.opengraph',
    'nbsphinx',
]

ogp_site_url = 'https://city2graph.net/'
#ogp_image = "https://city2graph.net/_images/city2graph_logo_main.png"
ogp_social_cards = {
    "image": "https://city2graph.net/_static/city2graph_logo.png",
}


templates_path = ['_templates']
exclude_patterns = ['_build']

language = 'en'

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'

# -- Options for numpydoc ----------------------------------------------------
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Set the favicon for the browser tab
html_favicon = '_static/city2graph_logo.png'

# Custom CSS
html_css_files = [
    'custom.css',
]

# Theme configuration
html_theme_options = {
    "github_url": "https://github.com/yu-ta-sato/city2graph",  # Replace with actual GitHub URL
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/city2graph_logo_wide.png",
        "image_dark": "_static/city2graph_logo_wide_dark.png",
    },
    "open_graph_image": "_static/city2graph_logo_wide.png",
    "open_graph_description": "A Python package for transforming urban data into graphs for Graph Neural Networks.",

}

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'momepy': ('http://docs.momepy.org/en/stable/', None),
}
