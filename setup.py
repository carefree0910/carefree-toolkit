import numpy
import platform
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

VERSION = "0.2.1"

DESCRIPTION = "Some commonly used functions and modules"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    "opencv-python",
    "dill",
    "future",
    "psutil",
    "pillow",
    "pathos",
    "cython>=0.29.12",
    "numpy>=1.16.2",
    "scipy>=1.2.1",
    "scikit-learn>=0.20.3",
    "matplotlib>=3.0.3",
]
if platform.system() != "Windows":
    INSTALL_REQUIRES.append("SharedArray")

setup(
    name="carefree-toolkit",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=INSTALL_REQUIRES,
    ext_modules=cythonize(
        Extension(
            "cftool.c.cython_utils",
            sources=["cftool/c/cython_utils.pyx"],
            language="c",
            include_dirs=[numpy.get_include(), "cftool/c"],
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ),
    package_data={"cftool.c": ["cython_utils.pyx"]},
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    url="https://github.com/carefree0910/carefree-toolkit",
    download_url=f"https://github.com/carefree0910/carefree-toolkit/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python numpy data-science",
)
