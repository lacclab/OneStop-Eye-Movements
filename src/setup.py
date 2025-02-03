from setuptools import setup, find_packages

setup(
    name="onestoptools",
    version="0.1.0",
    description=(
        "Julia and Python tools for linear mixed models"
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # TODO temporary, make sure inline with license in README
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    # set requirements
    # python requirement
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "juliacall",
    ],
)
