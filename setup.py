from setuptools import setup

from preppy import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=[__name__],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research'],
    pyython_requires='>=3.6.8',
    install_requires=['numpy'],
    url='https://github.com/phueb/Preppy',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Prepare language data for RNN training',
)