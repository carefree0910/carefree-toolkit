from setuptools import setup, find_packages

VERSION = "0.1.3"

DESCRIPTION = "Some commonly used functions and modules"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-toolkit",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pathos", "joblib",
        "dill", "future", "psutil", "pillow",
        "cython>=0.29.12", "numpy>=1.16.2", "scipy>=1.2.1",
        "scikit-learn>=0.20.3", "matplotlib>=3.0.3",
        "mkdocs", "mkdocs-material", "mkdocs-minify-plugin",
        "Pygments", "pymdown-extensions"
    ],
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    url="https://github.com/carefree0910/carefree-toolkit",
    download_url=f"https://github.com/carefree0910/carefree-toolkit/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python numpy data-science"
)
