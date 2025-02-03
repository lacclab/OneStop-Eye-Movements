from setuptools import setup, find_packages

setup(
    name="OneStop_linear_mm_tools",
    version="0.1.0",
    description=(
        "Julia and Python tools for linear mixed models"
    ),
    packages=find_packages(where="linear_mm_utils"),
    install_requires=[
        # add juliacall
        "juliacall",
    ],
)