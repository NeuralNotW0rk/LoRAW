from setuptools import setup, find_packages

setup(
    name='loraw',
    version='1.2.0',
    url='',
    author='Griffin Page',
    packages=find_packages(),    
    install_requires=[
        'stable_audio_tools',
        'bitsandbytes'
    ],
)