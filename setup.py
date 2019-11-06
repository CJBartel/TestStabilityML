#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:02:04 2019

@author: chrisbartel
"""

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='TestStabilityML',
        version='0.0.1',
        description='enabling formation energy models to be tested on stability',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/CederGroupHub/TestStabilityML',
        author=['Christopher J. Bartel'],
        author_email=['bartel.chrisj@gmail.com'],
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=[],
        test_suite='',
        tests_require=[],
        scripts=[]
    )
