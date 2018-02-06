from setuptools import setup

setup(
    name="neuroglancer-scripts",
    version="0.1",
    description="Conversion of images to the Neuroglancer pre-computed format",
    long_description="""\
neuroglancer-scripts
====================

Tools for conversion of images to the Neuroglancer pre-computed format.

`Repository on GitHub <https://github.com/HumanBrainProject/neuroglancer-scripts>`_
""",
    url="https://github.com/HumanBrainProject/neuroglancer-scripts",
    author="Yann Leprince",
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
        "test": ["coverage", "pytest"],
    },
    scripts=[
        "compute_scales.py",
        "convert_chunks.py",
        "generate_scales_info.py",
        "mesh_to_precomputed.py",
        "scale_stats.py",
        "slices_to_raw_chunks.py",
        "volume_to_raw_chunks.py",
    ],  # TODO convert to entry_points (below)
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
