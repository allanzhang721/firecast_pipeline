from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README.md if available
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8") if (this_dir / "README.md").exists() else ""

setup(
    name="firecast_pipeline",
    version="0.1.6",
    description="Unified regression pipeline for fire risk prediction using OLS, Lasso, MLP, CNN, and XGBoost.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Allan Zhang",
    author_email="yaoyu.zhang@mail.utoronto.ca",
    url="https://github.com/allanzhang721/firecast_pipeline",  # Optional: Update with actual repo
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.1.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "xgboost>=1.3.0",
        "torch>=1.9.0",
        "optuna>=2.3.0",
        "openpyxl>=3.0.0",
        "joblib>=1.0.0",
        "plotly>=5.0.0",
        "matplotlib>=3.3.0"  # Required for local plotting utilities
    ],
    entry_points={
        "console_scripts": [
            "firecast-train=regressorpipeline.train:main",
            "firecast-predict=regressorpipeline.predict:main",
            "firecast-visualize=regressorpipeline.visualize:main"
        ]
    },
    keywords=["fire risk", "regression", "CNN", "Optuna", "time to flashover", "fire safety"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
