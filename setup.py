#!/usr/bin/env python

import codecs
import os.path
import re
import sys

import setuptools


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Remember keep synchronized with the list of dependencies in tox.ini
tests_require = [
    "pytest",
    "requests-mock",
]

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []


setuptools.setup(
    name="neuroglancer-scripts",
    version=find_version("src", "neuroglancer_scripts", "__init__.py"),
    description="Conversion of images to the Neuroglancer pre-computed format",
    long_description=read("README.rst"),
    url="https://github.com/HumanBrainProject/neuroglancer-scripts",
    author="Yann Leprince",
    author_email="yann.leprince@cea.fr",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="neuroimaging",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=[
        "nibabel >= 2",
        "numpy >= 1.11.0",
        "pillow >= 1.1.6",
        "requests >= 2",
        "scikit-image",  # TODO use pillow instead
        "tqdm ~= 4.29",
        'imagecodecs', # required by scikit-image to read LZW compressed tiff files
    ],
    python_requires="~= 3.5",
    extras_require={
        "dev": tests_require + [
            "check-manifest",
            "flake8",
            "pep8-naming",
            "pre-commit",
            "pytest-cov",
            "readme_renderer",
            "sphinx",
            "tox",
        ],
    },
    setup_requires=pytest_runner,
    tests_require=tests_require,
    entry_points={
        "console_scripts": [
            "compute-scales="
            "neuroglancer_scripts.scripts.compute_scales:main",
            "convert-chunks="
            "neuroglancer_scripts.scripts.convert_chunks:main",
            "generate-scales-info="
            "neuroglancer_scripts.scripts.generate_scales_info:main",
            "link-mesh-fragments="
            "neuroglancer_scripts.scripts.link_mesh_fragments:main",
            "mesh-to-precomputed="
            "neuroglancer_scripts.scripts.mesh_to_precomputed:main",
            "scale-stats="
            "neuroglancer_scripts.scripts.scale_stats:main",
            "slices-to-precomputed="
            "neuroglancer_scripts.scripts.slices_to_precomputed:main",
            "volume-to-precomputed="
            "neuroglancer_scripts.scripts.volume_to_precomputed:main",
            "volume-to-precomputed-pyramid="
            "neuroglancer_scripts.scripts.volume_to_precomputed_pyramid:main",
        ],
    },
)
