import os
from setuptools import setup, find_packages

# Path to the requirements file
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Read the requirements from the requirements file without depending on pkg_resources
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
else:
    install_requires = []

setup(
    name="c2r-wan-infer",
    version="0.1.0",
    description="Inference-only WAN + DINO toolkit.",
    author="c2r",
    license="PolyForm-Noncommercial-1.0.0",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.12",
)
