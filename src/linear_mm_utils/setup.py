from setuptools import setup, find_packages

setup(
    name="OneStop_linear_mm_tools",
    version="0.1.0",
    description=(
        "Julia and Python tools for linear mixed models"
    ),
    classifiers=[
        'Development Status :: 1 - Beta',
        "Programming Language :: Python :: 3",
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        "License :: GPLv3",
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Software Development :: Libraries :: Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(where="linear_mm_utils"),
    install_requires=[
        # add juliacall
        "juliacall",
    ],
)
