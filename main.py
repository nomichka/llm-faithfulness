"""
Requires a config.ini file with the following format:
[openai]
api_key = sk-c...a
model = gpt-4
"""

# Imports
import configparser
from utils import *

# Get API key and model name from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config['openai']['api_key']
model = config['openai'].get('model', fallback='gpt-4')

# Steps 1, 2, 3
for rule in ["uppercase", "starts_with_a"]:
    print("Rule:", rule)
    print()

    accuracy, model_correctness = get_accuracy(rule, model=model)
    print("Accuracy using in-context:", accuracy)
    print("Model correctness on each test:", model_correctness)
    print()

    articulation = articulate(rule, model=model)
    print("Articulation:", articulation)
    print()

    faithful_fraction, model_faithful_correctness = test_faithfulness(rule, articulation, against_guess=model_correctness, model=model)
    print("Faithfulness using model-stated rule:", faithful_fraction)
    print("Model faithful correctness on each test:", model_faithful_correctness)

    failures = get_failures(rule, model_correctness, articulation=articulation, model_faithful_correctness=model_faithful_correctness)
    print(failures)
    print("-"*100)
    print()

    save_results(rule, accuracy, model_correctness, articulation, faithful_fraction, model_faithful_correctness, failures)

