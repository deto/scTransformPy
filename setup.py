import os
from setuptools import setup, find_packages

# Parse the version string
__version__ = ""
this_directory = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(this_directory, "scTransform", "_version.py")
exec(open(version_file).read())  # Loads version into __version__

setup(
    name="scTransform",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numba>=0.43.1',
        'numpy>=1.16.0',
        'pandas>=0.23.4',
        'patsy>=0.5.1',
        'scipy>=1.2.0',
        'statsmodels>=0.9.0',
        'tqdm>=4.29.1'
    ],

    author="David DeTomaso",
    author_email="davedeto@gmail.com",
    description="",
    keywords="",
    url=""
)
