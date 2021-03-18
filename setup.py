from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="osmm",
    version="0.0.1",
    packages=["osmm"],
    license="GPLv3",
    description="Oracle-structured minimization method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy >= 1.15",
        "scipy >= 1.1.0",
        "cvxpy >= 1.1.0a4"],
    url="https://github.com/cvxgrp/osmm",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)