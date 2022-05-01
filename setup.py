import numpy
import platform
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

VERSION = "0.2.8"

DESCRIPTION = "Some commonly used functions and modules"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    "tqdm",
    "opencv-python",
    "dill",
    "future",
    "psutil",
    "pillow",
    "pathos",
    "cython>=0.29.28",
    "numpy>=1.22.3",
    "scipy>=1.8.0",
    "scikit-learn>=1.0.2",
    "matplotlib>=3.5.1",
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
