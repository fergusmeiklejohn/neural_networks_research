from setuptools import find_packages, setup

setup(
    name="distribution-invention",
    version="0.1.0",
    author="Fergus Meiklejohn",
    description="Neural networks that invent new distributions",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "keras>=3.0",
        "torch>=2.0",
        "jax[metal]",
        "transformers",
        "wandb",
    ],
)
