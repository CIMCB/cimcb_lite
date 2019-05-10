from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cimcb_lite",
    version="1.0.0",
    description="A lite version of the cimcb package containing the necessary tools for the statistical analysis of untargeted and targeted metabolomics data.",
    long_description=long_description,
    license="MIT License",
    url="https://github.com/cimcb/cimcb_lite",
    packages=[
        "cimcb_lite",
        "cimcb_lite.bootstrap",
        "cimcb_lite.cross_val",
        "cimcb_lite.model",
        "cimcb_lite.plot",
        "cimcb_lite.utils"],
    python_requires='>=3.5',
    install_requires=[
        "bokeh>=1.0.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tqdm",
        "xlrd"],
    author='Kevin Mendez, David Broadhurst',
    author_email='k.mendez@ecu.edu.au, d.broadhurst@ecu.edu.au',
)
