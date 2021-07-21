from setuptools import setup, find_packages
# make sure numpy, numba, scipy and future are preexisting in the
# environent in which this is installed

setup(name='spectral_statistics_tools',
      version='1.1.1',
      description='The module for spectral statistics calculations.',
      url='https://github.com/JanSuntajs/spectral_statistics_tools',
      author='Jan Suntajs',
      author_email='Jan.Suntajs@ijs.si',
      license='MIT',
      packages=find_packages(),
      install_requires=[ ],  # 'numpy', 'scipy',
                             #'numba', 'future'],
      zip_safe=False)
