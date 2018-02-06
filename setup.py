#!/usr/bin/env python

from setuptools import setup

setup(
    name="neuroglancer-scripts",
    version="0.1",
    description="Conversion of images to the Neuroglancer pre-computed format",
    long_description="""\
neuroglancer-scripts
====================

Tools for conversion of images to the Neuroglancer pre-computed format.

`Documentation <http://neuroglancer-scripts.readthedocs.io/>`_

`Source code repository <https://github.com/HumanBrainProject/neuroglancer-scripts>`_

.. image:: https://readthedocs.org/projects/neuroglancer-scripts/badge/?version=latest
   :target: http://neuroglancer-scripts.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
""",
    url="https://github.com/HumanBrainProject/neuroglancer-scripts",
    author="Yann Leprince",
    author_email="y.leprince@fz-juelich.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="neuroimaging",
    package_dir={"": "src"},
    packages=["neuroglancer_scripts"],
    install_requires=[
        "nibabel >= 2",
        "numpy >= 1.11.0",
        "pillow >= 1.1.6",
        "requests >= 2",
        "tqdm >= 4"
    ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["pytest", "pytest-cov", "tox"],
    },
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "compute-scales=neuroglancer_scripts.scripts.compute_scales:main",
            "convert-chunks=neuroglancer_scripts.scripts.convert_chunks:main",
            "generate-scales-info=neuroglancer_scripts.scripts.generate_scales_info:main",
            "mesh-to-precomputed=neuroglancer_scripts.scripts.mesh_to_precomputed:main",
            "scale-stats=neuroglancer_scripts.scripts.scale_stats:main",
            "slices-to-precomputed=neuroglancer_scripts.scripts.slices_to_precomputed:main",
            "volume-to-precomputed=neuroglancer_scripts.scripts.volume_to_precomputed:main",
        ],
    },
)
