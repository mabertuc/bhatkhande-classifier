from setuptools import setup

setup(
    name='bhatkhande-classifier',
    version='0.0.1',
    packages=['classifier'],
    install_requires=[
        'tensorflow==2.4.0',
        'Keras==2.4.3',
        'click==7.1.2',
        'matplotlib==3.3.3',
        'music21==6.3.0',
        'opencv-python==4.4.0.46',
        'numpy==1.18.5'
    ],
    extras_require={
        'dev': [
            'pytest==6.1.2',
            'pylint==2.6.0',
            'coverage==5.3.1'
        ]
    },
    python_requires='==3.8.6'
)
