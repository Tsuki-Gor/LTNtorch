# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pytorch_sphinx_theme
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'LTNtorch'
copyright = '2021, Tommaso Carraro'
author = 'Tommaso Carraro'

# The full version, including alpha/beta/rc tags
release = '0.9'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages'
]

autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
    'python': ('https://docs.python.org/3', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# html_static_path = ['_static']
html_static_path = ['../../docs/_static'] # 更新路径

# 先生成latex，再生成PDF：生成Latex时，公式可能有错误，而且生成PDF的时候，有的公式好像无法正确显示，故放弃

# # -- 为了输出latex -------------------------------------------------
#
# # 如果没有以下部分，请添加
#
# # 主文档，指向入口文件
# master_doc = 'index'
#
# # LaTeX 配置
# latex_engine = 'pdflatex'
# latex_documents = [
#     (master_doc, 'ProjectName.tex', 'Project Name Documentation',
#      'Author Name', 'manual'),
# ]
#
# # 在 conf.py 文件中添加以下内容，以便在生成的 LaTeX 文件中包含所需的包，进而支持数学公式的显示
#
# latex_elements = {
#     'preamble': r'''
# \usepackage{amssymb}
# \usepackage{amsmath}
# '''
# }
