from setuptools import setup

setup(
    name="bhatkhande-classifier",
    version="0.0.1",
    packages=["classifier"],
    install_requires=[
        "cv2",
        "Numpy",
        "Keras",
        "Pillow",
        "tensorflow",
        "music21",
        "matplotlib"
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    }
)
