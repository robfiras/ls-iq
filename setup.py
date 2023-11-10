from setuptools import setup, find_packages

requires_list = ["mushroom_rl>=1.10.0", "tensorboard", "experiment-launcher"]

setup(name='imitation_lib',
      version='0.1',
      description='Code base of the paper: LS-IQ: Implicit Reward Regularization for Inverse Reinforcement Learning.',
      license='MIT',
      author="Firas Al-Hafez",
      packages=[package for package in find_packages()
                if package.startswith('imitation_lib')],
      install_requires=requires_list,
      zip_safe=False,
      )