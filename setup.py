import setuptools
from setuptools import find_packages
from distutils.core import setup

requirements = [
    'pyspark>=2.4.0',
]

setup(
    name='spark_utils',
    packages=find_packages(),
    version='1.0.0',
    install_requires=requirements,
    description='Spark utils.',
    author='Barak David'
)
