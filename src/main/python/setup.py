"""
PGL Python包配置文件
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pgl",
    version="1.0.0",
    author="CropAbility Team",
    author_email="team@cropability.com",
    description="Plant Genetics Lab - high performance GPU computing library",
    long_description="""
    PGL (Plant Genetics Lab): a high performance GPU computing library based on Triton.

    主要特性:
    - 基于Triton的GPU加速算子
    - TorchScript模型导出，支持Java/C++集成
    - 完整的性能基准测试和验证框架
    - 简单易用的Python API
    """,
    long_description_content_type="text/plain",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pgl=pgl_main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
