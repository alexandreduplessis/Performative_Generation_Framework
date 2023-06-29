from setuptools import setup


setup(name='perfgen',
      install_requires=[
            "numpy","torch", "pot", "tqdm", 'scipy', 'matplotlib',
            'scikit-learn', 'ipython', 'wandb', 'nflows'],
      packages=['perfgen'],
      )
