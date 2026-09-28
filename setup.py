from setuptools import setup, find_packages

setup(
    name="multilingual-entity-resolution",
    version="1.0.0",
    description="Hyper-Scale Multilingual Record Linkage under Capacitated Bipartite Constraints",
    author="Siva Rama Krishna Reddy Padala",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "rapidfuzz>=3.0.0",
        "tqdm>=4.65.0",
    ],
)
