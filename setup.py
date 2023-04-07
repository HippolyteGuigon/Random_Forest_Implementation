from setuptools import setup, find_packages

setup(
    name="Random_Forest_Implementation",
    version="0.1.0",
    packages=find_packages(
        include=["Random_forest", "Random_forest.*"]
    ),
    description="Python implementation of the Random Forest Machine\
        Learning Algorithm",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Random_Forest_Implementation",
)