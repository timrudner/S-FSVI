import codecs
from setuptools import setup, find_packages

with codecs.open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="fsvi",
    description="Function-Space Variational Inference",
    long_description=README,
    long_description_content_type="text/markdown",
    version="0.1",
    packages=find_packages(),
    author="Tim Rudner",
    author_email="tim.rudner@cs.ox.ac.uk",
    python_requires=">=3.7",
    entry_points={"console_scripts": ["fsvi=cli:cli"]},
)
