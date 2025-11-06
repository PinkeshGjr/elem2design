from setuptools import setup, find_packages

# Find all packages including common and crello
packages = find_packages(include=['common', 'common.*', 'crello', 'crello.*'])

setup(
    name="dataset",
    version="0.0.1",
    packages=packages,
    python_requires=">=3.10",
)
