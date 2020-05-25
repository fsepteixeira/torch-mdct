from setuptools import find_packages, setup

NAME = 'torch_mcdt'
REQUIREMENTS = [
    'numpy',
    'torch',
    'scipy',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

EXTRAS={
        'tests': ['pytest']
    }

setup(
    name="torch_mdct",
    version="0.0.0",
    description="Pytorch implementation of MDCT/iMDCT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fsept11/torch-mdct",
    author="Francisco Teixeira",
    author_email="francisco.s.teixeira@tecnico.ulisboa.pt",
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Signal Processing"
    ],
    # Exclude the build files.
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_requires=EXTRAS,
)
