#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smart",
    version="0.0.1",
    author="Chih-Chun Hsu",
    author_email="chh194@ucsd.edu",
    packages=setuptools.find_packages(),
    url="https://github.com/chihchunhsu/smart",
    license="MIT",
    description=("The Spectral Modeling Analysis and RV Tool"),
    long_description=open("README.md").read(),
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=False,
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "astropy", "emcee", "corner"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=True,
)
