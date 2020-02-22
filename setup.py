from setuptools import setup, find_packages

requires = ["numpy", "pandas", "SQLAlchemy", "scikit-learn", "teradata",
            "sqlalchemy-teradata", "teradatasql", "teradatasqlalchemy",
            "paramiko", "scp"]

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup(
    name="sklearn2vantage",
    version="0.1.6",
    description="Module for converting sklearn model to Teradata Vantage"
                + " model",
    long_description=readme,
    url="https://github.com/Yukimura66/sklearn2vantage",
    author="Akihiro Sanada",
    author_email="akihiro.sanada@icloud.com",
    license="MIT",
    keywords="Teradata scikit-learn Vantage",
    packages=find_packages(),
    install_requires=requires,
    classifiers=[
        "Topic :: Database",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
