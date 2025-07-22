#!/usr/bin/env python3
"""Fix for ModificationPair format issue in training script"""

def convert_modification_pairs_to_training_format(modification_pairs):
    """
    Convert ModificationPair objects to the dictionary format expected by training.
    
    The training script expects dictionaries with keys:
    - command: the input command
    - action: the output action sequence  
    - modification: description of the modification
    - original_action: the original action sequence (for reference)
    - modified_action: the modified action sequence
    """
    converted = []
    
    for pair in modification_pairs:
        # Use the modified sample as the training example
        training_sample = {
            'command': pair.modified_sample.command,
            'action': pair.modified_sample.action,
            'modification': pair.modification_description,
            'original_action': pair.original_sample.action,
            'modified_action': pair.modified_sample.action,
            'modification_type': pair.modification_type,
            'modification_rules': pair.modification_rules
        }
        converted.append(training_sample)
    
    return converted

# Example fix to add to paperspace_train_with_safeguards.py:
"""
# After loading modifications:
mod_generator = ModificationGenerator()
modification_pairs = mod_generator.load_modifications()

# Convert to training format
modifications = []
for pair in modification_pairs:
    # Create training sample from modified version
    sample = {
        'command': pair.modified_sample.command,
        'action': pair.modified_sample.action,
        'modification': pair.modification_description,
    }
    modifications.append(sample)
"""