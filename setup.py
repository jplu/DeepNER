# coding=utf-8

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow_gpu>=1.13.0',
                     'pandas>=0.23.4']

setup(
    name='deepner',
    version='1.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='DeepNER'
)