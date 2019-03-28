from setuptools import setup, Extension, distutils, Command, find_packages

if __name__ == '__main__':
    setup(
        name="BinTorch",
        version='0.1',
        description=("Simple replication of PyTorch"),
        install_requires=['jax', 'jaxlib']
        )
