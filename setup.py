from setuptools import setup, find_namespace_packages

setup(name='samplify',
      packages=find_namespace_packages(include=["samplify", "samplify.*"]),
      version='0.1',
      description='none',
      url='127.0.0.1',
      author_email='karol.gotkowski@dkfz.de',
      license='private - atm',
      install_requires=[
            "numpy>=1.20",
            "scikit-image",
            "scipy",
            "tqdm",
            'zarr'
      ],
      zip_safe=False)
