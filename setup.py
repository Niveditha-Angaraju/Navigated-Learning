from setuptools import setup, find_packages

setup(
    name="my_gym_env",
    version="0.0.1",
    install_requires=["gym", "pygame", "numpy"],
    packages=find_packages(),  # Automatically finds all packages
)

