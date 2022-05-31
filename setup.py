#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:21:59 2022

@author: Dean Thomas
"""

from setuptools import setup, find_packages

# Numba typically requires an older version of numpy. So put first.
install_requires = [
                        "numba",
                        "pandas",
                        "numpy",
                        "joblib",
                        "matplotlib"
                    ]

setup(
    name='magfli',
    version='0.0.1',
    author='Dean Thomas',
    author_email='dthoma6@gmu.edu',
    packages=find_packages(),
    description='Trace magnetic field lines given a magnetic field',
    install_requires=install_requires
)
