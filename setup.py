from setuptools import setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    name="cimcb_lite",
    version="1.0.1",
    description="A lite version of the cimcb package containing the necessary tools for the statistical analysis of untargeted and targeted metabolomics data.",
    long_description=readme(),
    long_description_content_type='text/markdown',
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
