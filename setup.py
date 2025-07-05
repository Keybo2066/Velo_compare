#!/usr/bin/env python3
"""
WTKO RNA Velocity + Contrastive Learning Pipeline
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file"""
    req_file = os.path.join(here, 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]
    return []

# Package metadata
setup(
    # Basic package information
    name="wtko-pipeline",
    version="1.0.0",
    description="RNA Velocity analysis with contrastive learning for WT/KO single-cell comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Kohei Kubo",  # Replace with your actual name
    author_email="ctmk0009@mail4.doshisha.ac.jp",  # Replace with your email
    
    # URLs
    url="https://github.com/Keybo2066/Velo_compare",  # Replace with your GitHub URL
    project_urls={
        "Bug Reports": "https://github.com/Keybo2066/Velo_compare/issues",
        "Source": "https://github.com/Keybo2066/Velo_compare",
        "Documentation": "https://github.com/Keybo2066/Velo_compare#readme",
    },
    
    # Package discovery
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "tutorial": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "nbconvert>=6.0.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords for package discovery
    keywords="single-cell RNA-seq velocity contrastive-learning bioinformatics genomics",
    
    # Entry points (command line scripts)
    entry_points={
        "console_scripts": [
        ],
    },
    
    
    # Zip safety
    zip_safe=False,
    
    # Platform specification
    platforms=["any"],
)
