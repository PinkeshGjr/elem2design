from setuptools import setup, find_packages

setup(
    name="opencole",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "skia-python>=87.5",
        "datasets>=3.2.0",
        "langchain-core>=0.3.28",
        "langchain>=0.3.13",
        "pydantic>=2.10.0",
    ],
)
