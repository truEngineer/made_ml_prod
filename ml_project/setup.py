from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Homework1",
    author="TruEngineer",
    install_requires=[
        "marshmallow-dataclass==8.4.1",
        "pandas==1.2.4",
        "pandas-profiling==2.11.0",
        "pytest==6.2.3",
        "scikit-learn==0.24.1",
        "numpy==1.20.1",
        "pyyaml==5.4.1",
        "click==7.1.2",
        "matplotlib==3.3.4",
        "seaborn==0.11.1",
    ],
    license="MIT",
)