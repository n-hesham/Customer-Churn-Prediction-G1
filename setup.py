```python
from setuptools import setup, find_packages

setup(
    name="customer-churn-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'fastapi',
        'uvicorn',
        'pyyaml',
        'scipy',
        'psutil',
        'pytest',
        'seaborn',
        'matplotlib'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Customer Churn Prediction with MLOps and Monitoring",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/customer-churn-prediction",
)
```