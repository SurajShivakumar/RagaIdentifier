"""
Raga definitions with arohanam (ascending) and avarohanam (descending) patterns.
Each raga is defined by its characteristic note sequences.
"""

# Note mapping: Sa Re Ga Ma Pa Dha Ni
# Using Western note names for easier processing
# S = Sa (C), R = Re, G = Ga, M = Ma, P = Pa, D = Dha, N = Ni
# With variants: R1, R2, R3, G1, G2, G3, M1, M2, D1, D2, D3, N1, N2, N3

RAGA_PATTERNS = {
    "Mayamalavagowla": {
        "arohanam": ["S", "R1", "G3", "M1", "P", "D1", "N3", "S'"],
        "avarohanam": ["S'", "N3", "D1", "P", "M1", "G3", "R1", "S"],
        "characteristic_phrases": [
            ["S", "R1", "G3", "M1"],
            ["G3", "M1", "P", "D1"],
            ["N3", "D1", "P"],
            ["R1", "G3"]
        ],
        "important_notes": ["R1", "G3", "D1", "N3"],
        "vadi": "D1",  # Most important note
        "samvadi": "G3",  # Second most important note
        "swara_frequencies": {
            "S": 261.63,   # Sa - C
            "R1": 275.00,  # Shuddha Rishabham - Db (komal re)
            "G3": 329.63,  # Antara Gandharam - E
            "M1": 349.23,  # Shuddha Madhyamam - F
            "P": 392.00,   # Panchamam - G
            "D1": 412.50,  # Shuddha Dhaivatam - Ab (komal dha)
            "N3": 493.88,  # Kakali Nishadam - B
            "S'": 523.25   # Upper Sa - C'
        }
    },
    "Shankarabharanam": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D2", "N3", "S'"],
        "avarohanam": ["S'", "N3", "D2", "P", "M1", "G3", "R2", "S"],
        "characteristic_phrases": [
            ["S", "R2", "G3", "M1"],
            ["P", "D2", "N3", "S'"],
            ["G3", "R2", "S"],
            ["M1", "P", "D2"]
        ],
        "important_notes": ["R2", "G3", "D2", "N3"],
        "vadi": "D2",
        "samvadi": "G3",
        "swara_frequencies": {
            "S": 261.63,   # Sa - C
            "R2": 293.66,  # Chatushruti Rishabham - D
            "G3": 329.63,  # Antara Gandharam - E
            "M1": 349.23,  # Shuddha Madhyamam - F
            "P": 392.00,   # Panchamam - G
            "D2": 440.00,  # Chatushruti Dhaivatam - A
            "N3": 493.88,  # Kakali Nishadam - B
            "S'": 523.25   # Upper Sa - C'
        }
    }
}

def get_raga_info(raga_name):
    """Get complete information about a raga."""
    return RAGA_PATTERNS.get(raga_name, None)

def get_all_ragas():
    """Get list of all available ragas."""
    return list(RAGA_PATTERNS.keys())
