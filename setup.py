from setuptools import setup

setup(
    name="bhatkhande-classifier",
    version="0.0.1",
    packages=["classifier"],
    install_requires=[
        "Keras",
        "Pillow",
        "tensorflow",
        "music21"
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    }
)
