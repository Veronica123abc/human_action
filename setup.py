from setuptools import setup, find_packages

setup(
    name="human_action",
    version="0.1.0",
    description="3D representation of human actions from video, images, or image sequences",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "human-action=main:main",
        ],
    },
)
