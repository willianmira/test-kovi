from setuptools import setup, find_packages

setup(
    name="churn-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.13.0",
        "dash>=2.8.0",
        "shap>=0.41.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "fastapi>=0.95.2",
        "uvicorn>=0.23.2",
        "python-multipart>=0.0.6",
        "joblib>=1.3.0",
    ],
    author="Willian Mira",
    author_email="seu.email@exemplo.com",
    description="Análise completa de churn para motoristas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 