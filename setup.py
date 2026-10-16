#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
import setuptools

#import smart
#from smart.__init__ import __version__

# PyPI
with open("smart/__init__.py") as f:
    __version__ = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read()).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smart-astro",
    version=__version__,
    author="Chih-Chun Hsu",
    author_email="chsu@northwestern.edu",
    packages=setuptools.find_packages(),
    url="https://github.com/chihchunhsu/smart",
    license="MIT",
    description=("The Spectral Modeling Analysis and RV Tool"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown", # for PyPI
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=False,
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "astropy", "emcee", "corner"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False,
)
