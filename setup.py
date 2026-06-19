"""screenkb package setup."""

from setuptools import setup, find_packages

setup(
    name="screenkb",
    version="2.0.0",
    description="Screenshot analysis CLI — OCR + layout + knowledge base → structured JSON",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0",
        "Pillow>=10.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.8",
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "screenkb=screenkb.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
