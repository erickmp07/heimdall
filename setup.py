from setuptools import setup, find_packages

with open('README.md') as file:
    readme = file.read()

with open('LICENSE') as file:
    license = file.read()

setup(
    name='heimdall',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Erick Macedo Pinto',
    author_email='erimacedo_92@hotmail.com',
    url='https://github.com/erickmp07/heimdall',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)