from setuptools import setup


setup(name='perfgen',
      install_requires=[
            "numpy","torch", "ot", "sklearn", "tqdm", 'scipy>=0.18.0', 'matplotlib>=2.0.0',
            'scikit-learn>=1.0', 'pandas', 'ipython'],
      packages=['perfgen'],
      )
