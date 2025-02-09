from setuptools import setup, find_packages

setup(
    name="jax_autodiff",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "graphviz>=0.20.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Surya Subramanian",
    description="A JAX-inspired automatic differentiation compiler with PyTorch backend and Metal acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
