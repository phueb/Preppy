from setuptools import setup


setup(
    name='preppy',
    version='3.0.0',  # keep version here, otherwise import error during install
    packages=['preppy'],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research'],
    pyython_requires='>=3.7',
    install_requires=[
        'numpy==1.18.1',
        'cached_property',
        'sortedcontainers',
        'cached_property',
    ],
    url='https://github.com/phueb/Preppy',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Prepare language data for RNN training',
)