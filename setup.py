from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brainforge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="左右脑协作的高效LLM微调系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/brainforge",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/brainforge/issues",
        "Documentation": "https://brainforge.readthedocs.io",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "cohere>=4.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.5",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "brainforge=brainforge.cli.commands:cli",
        ],
    },
)