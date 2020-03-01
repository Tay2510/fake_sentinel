from setuptools import setup, find_packages

import fake_sentinel

setup(
    name='fake_sentinel',
    author='Jeremy Yen',
    author_email='tay2510@gmail.com',
    version=fake_sentinel.__version__,
    description='Custom package for Kaggle DFDC competition',
    packages=find_packages(),
)
