from setuptools import setup


setup(
    name='preppy',
    version='2.0.1',
    packages=['preppy'],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research'],
    pyython_requires='>=3.6.8',
    install_requires=[
        'numpy==1.17.3',
        'tokenizers==0.10.1',
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