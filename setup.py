from setuptools import setup, find_packages
import pip


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

pkgs = find_packages(exclude=('examples', 'docs'))

# absolutely ridiculous manual install of pyswarm-0.7+ from GitHub
# this should be taken care of via dependency_links below, but it
# does. not. work.
pip.main(['install', '--upgrade', 'git+https://github.com/smcolby/pyswarm@passpool'])

setup(
    name='nmrfit',
    version='0.1',
    description='Quantitative NMR package.',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/smcolby/nmrfit',
    license=license,
    packages=pkgs,
    install_requires=required,
    dependency_links=[
        "git+https://github.com/smcolby/pyswarm@passpool",
    ]
)
