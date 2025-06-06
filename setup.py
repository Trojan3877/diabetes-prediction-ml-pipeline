diabetes-prediction-ml-pipeline/setup.py

from setuptools import setup, find_packages

setup(
    name="diabetes_prediction_ml_pipeline",
    version="0.1.0",
    description="A modular ML pipeline for diabetes prediction (Logistic Regression & Random Forest)",
    author="Corey Leath (Trojan3877)",
    author_email="your.email@example.com",
    url="https://github.com/Trojan3877/diabetes-prediction-ml-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "diabetes-train=src.train:main",
            "diabetes-evaluate=src.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
