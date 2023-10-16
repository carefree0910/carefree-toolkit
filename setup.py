from setuptools import setup
from setuptools import find_packages


VERSION = "0.3.9"

DESCRIPTION = "Some commonly used functions and modules"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    "dill",
    "tqdm",
    "future",
    "psutil",
    "pathos",
    "pydantic",
    "numpy>=1.22.3",
]

setup(
    name="carefree-toolkit",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=INSTALL_REQUIRES,
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-toolkit",
    download_url=f"https://github.com/carefree0910/carefree-toolkit/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python numpy data-science",
)
