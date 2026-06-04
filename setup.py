from setuptools import setup, find_packages

setup(
    name="complex-network-cascading-failure",
    version="0.1.0",
    author="研0新生",
    description="复杂网络级联失效仿真平台",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scipy>=1.10",
    ],
)
