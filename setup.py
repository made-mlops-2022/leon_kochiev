from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="mlops_hw1",
    packages=find_packages(),
    version="0.1.0",
    description="Heart cleveland dataset research",
    author="Leon Kochiev",
    entry_points={
        "console_scripts": [
            "ml_train = src.utils.train:train"
        ]
    },
    install_requires=required,
)