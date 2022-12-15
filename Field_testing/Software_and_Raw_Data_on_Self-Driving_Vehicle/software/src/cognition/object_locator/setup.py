#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['zzz_cognition_object_locator'],
    package_dir={'': 'src'}
)

setup(**d)