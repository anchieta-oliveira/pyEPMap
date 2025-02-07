from setuptools import setup, find_packages

def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="pyepm",
    version="0.0.1",
    description="The pyEPMaps (Python Electrostatic Potential Maps) is a Python library designed for the calculation and visualization of Electrostatic Potential Maps (EPM), which can be accelerated by GPU.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author_email="anchieta.oliveira@biof.ufrj.br",
    url="https://github.com/anchieta-oliveira/pyEPMaps",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "sphinx>=8.1.3",
        "numba>=0.60.0",
        "numpy>=2.0.2",
        "scipy>=1.15.1",
        "cupy>=13.3.0",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
