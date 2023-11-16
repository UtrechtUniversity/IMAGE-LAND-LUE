# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sphinx_rtd_theme
# import lue

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../imagelcm'))
sys.path.insert(0, os.path.abspath('C:\\Users\\2331764\\AppData\\Local\\anaconda3\\pkgs\\lue-0.3.7-py311h30d3044_1\\Lib\\site-packages\\lue'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ImageLCM'
copyright = '2023, Ben Romero-Wilcock'
author = 'Ben Romero-Wilcock'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon',
              "sphinx.ext.autodoc",
              'sphinx_rtd_theme',
              'myst_parser']
# extensions = ['sphinx.ext.napoleon', "autoapi.extension", 'sphinx_rtd_theme']
# autoapi_dirs = ['../../imagelcm']

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "linkify"
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']

# html_theme_options = {'style_nav_header_background': '#f759ea'}
html_theme_options = {'style_external_links': True}

napoleon_use_param = True