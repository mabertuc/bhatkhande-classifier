"""Constants."""
CHARACTER_TO_NOTE_MAPPING = {
    'flat-seven-low': 'B-4',
    'seven-low': 'B4',
    'one': 'C4',
    'flat-two': 'D-4',
    'two': 'D4',
    'flat-three': 'E-4',
    'three': 'E4',
    'four': 'F4',
    'sharp-four': 'F#4',
    'five': 'G4',
    'flat-six': 'A-4',
    'six': 'A4',
    'flat-seven': 'B-4',
    'seven': 'B4',
    'one-high': 'C5',
    'flat-two-high': 'D-5',
    'two-high': 'D5',
    'three-high': 'E5',
    'five-high': 'G5',
}
NOTE_CHARACTERS = list(CHARACTER_TO_NOTE_MAPPING.keys())
NON_NOTE_CHARACTERS = [
    'divider',
    'continue',
    'rest',
    'rhythm-o',
    'rhythm-three',
    'rhythm-two',
    'rhythm-x'
]
VALID_CLASS_NAMES = NOTE_CHARACTERS + NON_NOTE_CHARACTERS
VALID_IMAGE_FILE_EXTENSION = '.png'
TARGET_IMAGE_HEIGHT = 30
TARGET_IMAGE_WIDTH = 30
