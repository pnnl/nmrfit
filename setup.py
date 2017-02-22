from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nmrfit',
    version='0.1',
    description='Quantitative NMR package.',
    long_description=readme,
    author='Sean Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/smcolby/nmrfit',
    license=license,
    packages=find_packages(exclude=('data', 'docs')),
    dependency_links=[
        "git+https://github.com/tisimst/pyswarm#egg=pyswarm"
    ]
)
