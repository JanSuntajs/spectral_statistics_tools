from setuptools import setup, find_packages


setup(name='spectral_statistics_tools',
      version='1.0.3',
      description='The module for spectral statistics calculations.',
      url='https://github.com/JanSuntajs/spectral_statistics_tools',
      author='Jan Suntajs',
      author_email='Jan.Suntajs@ijs.si',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy',
                        'numba', 'future', 'h5py==2.10.0',
                        'pandas'],
      zip_safe=False)
