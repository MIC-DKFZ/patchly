from setuptools import setup, find_namespace_packages

setup(name='patchly',
      packages=find_namespace_packages(include=["patchly", "patchly.*"]),
      version='0.1',
      description='none',
      url='',
      author_email='karol.gotkowski@dkfz.de',
      license='Apache Software License 2.0',
      install_requires=[
            "numpy>=1.20"
      ],
      zip_safe=False)
