from setuptools import setup, find_packages

setup(
    name="dental-vision",
    version="0.1.0",
    description="Deep Learning Suite for Panoramic Dental X-ray Tooth Localization & Pathology Detection",
    author="Dental Vision Contributors",
    author_email="contact@dentalvision.org",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "dental-vision=main:main",
        ],
    },
)
