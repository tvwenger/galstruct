from setuptools import setup

setup(
    name="galstruct",
    version="0.1.dev0",
    description="Modeling Galactic structure",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["galstruct"],
    install_requires=["numpy", "torch", "matplotlib"],
)
