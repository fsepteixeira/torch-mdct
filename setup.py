from setuptools import find_packages, setup

NAME = 'torch_mdct'
REQUIREMENTS = [
    'numpy',
    'torch',
    'torchaudio',
    'scipy',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch_mdct",
    version="0.0.1",
    description="Pytorch implementation of MDCT/iMDCT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fsepteixeira/torch-mdct",
    author="Francisco Teixeira",
    author_email="francisco.s.teixeira@tecnico.ulisboa.pt",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
