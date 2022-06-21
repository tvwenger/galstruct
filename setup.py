from setuptools import setup, find_packages

setup(
    name="galstruct",
    version="0.1.dev1",
    description="Modeling Galactic structure",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "torch", "matplotlib", "sbi", "dill", "pymc"],
)
