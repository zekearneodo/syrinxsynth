from setuptools import setup

setup(name='syrinxsynth',
      version='0.1',
      description='Tools to generate synthetic vocalizations using the Mindlin model',
      url='http://github.com/zekearneodo/syrinxsynth',
      author='Zeke Arneodo',
      author_email='ezequiel@ini.ethz.ch',
      license='MIT',
      packages=['syrinxsynth'],
      install_requires=['numpy',
                        'numba'],
      zip_safe=False)