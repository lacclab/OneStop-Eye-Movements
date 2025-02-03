from setuptools import setup, find_packages

setup(
    name="OneStopTools",
    version="0.1.0",
    description=(
        "Julia and Python tools for linear mixed models"
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # TODO temporary, make sure inline with license in README
        "Operating System :: OS Independent",
    ],
    py_modules=["calc_means_main", "constants", "julia_linear_mm", "utils"],

    # packages=find_packages(where="linear_mm_utils"),
    # install_requires=[
    #     # add juliacall
    #     "juliacall",
    # ],
    
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
