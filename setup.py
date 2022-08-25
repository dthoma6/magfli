#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:21:59 2022

@author: Dean Thomas
"""

from setuptools import setup, find_packages

install_requires = [
                        "pandas",
                        "numpy",
                        "matplotlib",
                        "swmfio",
                        "vtk"
                    ]

setup(
    name='magfli',
    version='0.8.0',
    author='Dean Thomas',
    author_email='dthoma6@gmu.edu',
    packages=find_packages(),
    description='Trace field lines given a field',
    install_requires=install_requires
)
