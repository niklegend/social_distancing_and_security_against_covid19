import os

from setuptools import setup, find_packages

here = os.path.dirname(__file__)

setup(
    name='masterthesis',
    version='0.0.2',
    author='Mattia Vandi',
    author_email='mattia.vandi@studio.unibo.it',
    packages=find_packages(here),
    python_requires='>=3.6',
    install_requires=[
        # 'boto3',
        # 'botocore',
        'numpy',
        'pandas',
        'pillow',
        # 'pycocotools',
        'scikit-learn',
        'torch>=1.6.0',
        'torchvision>=0.5.0',
        'tqdm',
        'typing_extensions',
        'opencv-python',
        'bbox-visualizer',
        'colormap',
        'easydev'
    ]
)
