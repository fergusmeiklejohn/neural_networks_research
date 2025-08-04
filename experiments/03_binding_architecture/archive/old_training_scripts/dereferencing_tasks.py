"""
Dereferencing Tasks for Variable Binding

These tasks force the model to perform explicit variable binding by requiring
it to dereference variables to their bound meanings. This is the key innovation
from Wu et al. (2025) that enables true compositional generalization.

Types of tasks:
1. Simple binding: "X means jump. Do X." -> "JUMP"
2. Multiple bindings: "X means jump. Y means walk. Do X then Y." -> "JUMP WALK"
3. Rebinding: "X means jump. Do X. Now X means walk. Do X." -> "JUMP WALK"
4. Compositional: "X means jump. Do X twice." -> "JUMP JUMP"
"""

from utils.imports import setup_project_paths
setup_project_paths()

import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)


class DereferencingTaskGenerator:
    """Generate tasks that require variable binding and dereferencing."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Define vocabularies
        self.word_vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
            # Variable names
            'X': 4, 'Y': 5, 'Z': 6, 'A': 7, 'B': 8,
            # Actions
            'jump': 9, 'walk': 10, 'run': 11, 'turn': 12, 'look': 13,
            # Modifiers
            'left': 14, 'right': 15, 'twice': 16, 'thrice': 17,
            # Meta words
            'means': 18, 'do': 19, 'now': 20, 'and': 21, 'then': 22,
            # Numbers
            'two': 23, 'three': 24, 'times': 25
        }
        
        self.action_vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'JUMP': 3, 'WALK': 4, 'RUN': 5, 'TURN': 6, 'LOOK': 7,
            'LEFT': 8, 'RIGHT': 9
        }
        
        # Reverse mappings
        self.word_to_id = self.word_vocab
        self.id_to_word = {v: k for k, v in self.word_vocab.items()}
        self.action_to_id = self.action_vocab
        self.id_to_action = {v: k for k, v in self.action_vocab.items()}
        
        # Variable names and action words
        self.variables = ['X', 'Y', 'Z', 'A', 'B']
        self.actions = ['jump', 'walk', 'run', 'turn', 'look']
        self.modifiers = ['left', 'right']
        
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Convert words to token IDs."""
        return np.array([self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in words])
    
    def encode_actions(self, actions: List[str]) -> np.ndarray:
        """Convert actions to token IDs."""
        return np.array([self.action_to_id.get(a, self.action_to_id['<PAD>']) for a in actions])
    
    def generate_simple_binding(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Generate simple binding task: "X means jump. Do X." -> "JUMP"
        
        Returns:
            command: List of words
            actions: List of action tokens
            bindings: Dict of variable bindings
        """
        var = random.choice(self.variables)
        action = random.choice(self.actions)
        
        command = [var, 'means', action, '.', 'do', var, '.']
        actions = [action.upper()]
        bindings = {var: action}
        
        return command, actions, bindings
    
    def generate_multiple_bindings(self, n_vars: int = 2) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Generate task with multiple variable bindings.
        "X means jump. Y means walk. Do X then Y." -> "JUMP WALK"
        """
        vars_used = random.sample(self.variables, n_vars)
        actions_used = random.sample(self.actions, n_vars)
        
        command = []
        bindings = {}
        
        # Define bindings
        for var, action in zip(vars_used, actions_used):
            command.extend([var, 'means', action, '.'])
            bindings[var] = action
        
        # Create execution sequence
        command.append('do')
        execution_order = random.sample(vars_used, len(vars_used))
        
        for i, var in enumerate(execution_order):
            command.append(var)
            if i < len(execution_order) - 1:
                command.append('then')
        command.append('.')
        
        # Generate expected actions
        actions = [bindings[var].upper() for var in execution_order]
        
        return command, actions, bindings
    
    def generate_rebinding(self) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Generate rebinding task where variable meaning changes.
        "X means jump. Do X. Now X means walk. Do X." -> "JUMP WALK"
        """
        var = random.choice(self.variables)
        action1, action2 = random.sample(self.actions, 2)
        
        command = [
            var, 'means', action1, '.', 'do', var, '.',
            'now', var, 'means', action2, '.', 'do', var, '.'
        ]
        
        actions = [action1.upper(), action2.upper()]
        
        # Track binding history
        bindings = {var: [action1, action2]}
        
        return command, actions, bindings
    
    def generate_compositional(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Generate compositional task requiring binding + modification.
        "X means jump. Do X twice." -> "JUMP JUMP"
        """
        var = random.choice(self.variables)
        action = random.choice(self.actions)
        
        # Choose repetition
        rep_word, rep_count = random.choice([('twice', 2), ('thrice', 3)])
        
        command = [var, 'means', action, '.', 'do', var, rep_word, '.']
        actions = [action.upper()] * rep_count
        bindings = {var: action}
        
        return command, actions, bindings
    
    def generate_with_modifiers(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Generate task with directional modifiers.
        "X means turn. Do X left." -> "TURN LEFT"
        """
        var = random.choice(self.variables)
        action = random.choice(['turn', 'look'])  # Actions that take modifiers
        modifier = random.choice(self.modifiers)
        
        command = [var, 'means', action, '.', 'do', var, modifier, '.']
        actions = [action.upper(), modifier.upper()]
        bindings = {var: action}
        
        return command, actions, bindings
    
    def generate_dataset(
        self, 
        n_samples: int = 1000,
        task_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate a dataset of dereferencing tasks.
        
        Args:
            n_samples: Number of samples to generate
            task_distribution: Distribution of task types
            
        Returns:
            Dictionary with train/val/test splits
        """
        if task_distribution is None:
            task_distribution = {
                'simple': 0.3,
                'multiple': 0.2,
                'rebinding': 0.2,
                'compositional': 0.2,
                'modifiers': 0.1
            }
        
        # Generate tasks
        tasks = []
        
        for task_type, proportion in task_distribution.items():
            n_tasks = int(n_samples * proportion)
            
            for _ in range(n_tasks):
                if task_type == 'simple':
                    command, actions, bindings = self.generate_simple_binding()
                elif task_type == 'multiple':
                    command, actions, bindings = self.generate_multiple_bindings()
                elif task_type == 'rebinding':
                    command, actions, bindings = self.generate_rebinding()
                elif task_type == 'compositional':
                    command, actions, bindings = self.generate_compositional()
                elif task_type == 'modifiers':
                    command, actions, bindings = self.generate_with_modifiers()
                
                # Remove periods for cleaner encoding
                command = [w for w in command if w != '.']
                
                tasks.append({
                    'command': command,
                    'actions': actions,
                    'bindings': bindings,
                    'task_type': task_type
                })
        
        # Shuffle tasks
        random.shuffle(tasks)
        
        # Split into train/val/test (70/15/15)
        n_train = int(0.7 * len(tasks))
        n_val = int(0.15 * len(tasks))
        
        train_tasks = tasks[:n_train]
        val_tasks = tasks[n_train:n_train + n_val]
        test_tasks = tasks[n_train + n_val:]
        
        # Convert to arrays
        dataset = {
            'train': self._tasks_to_arrays(train_tasks),
            'val': self._tasks_to_arrays(val_tasks),
            'test': self._tasks_to_arrays(test_tasks)
        }
        
        return dataset
    
    def _tasks_to_arrays(self, tasks: List[Dict]) -> Dict[str, np.ndarray]:
        """Convert task list to numpy arrays."""
        max_command_len = max(len(t['command']) for t in tasks)
        max_action_len = max(len(t['actions']) for t in tasks)
        
        n_tasks = len(tasks)
        commands = np.zeros((n_tasks, max_command_len), dtype=np.int32)
        actions = np.zeros((n_tasks, max_action_len), dtype=np.int32)
        
        for i, task in enumerate(tasks):
            cmd_encoded = self.encode_words(task['command'])
            act_encoded = self.encode_actions(task['actions'])
            
            commands[i, :len(cmd_encoded)] = cmd_encoded
            actions[i, :len(act_encoded)] = act_encoded
        
        return {
            'commands': commands,
            'actions': actions,
            'metadata': tasks  # Keep original data for analysis
        }
    
    def generate_modification_test_set(self) -> Dict[str, np.ndarray]:
        """
        Generate test set specifically for modification capability.
        
        This tests if the model can modify bindings on the fly,
        which is the key capability we're trying to achieve.
        """
        test_cases = []
        
        # Test 1: Simple modification (jump -> hop)
        # Train on "X means jump", test on "X means hop"
        for var in self.variables[:2]:  # Use only X and Y for test
            # Original binding
            orig_command = [var, 'means', 'jump', 'do', var]
            orig_actions = ['JUMP']
            
            # Modified binding (not in training vocab)
            mod_command = [var, 'means', 'walk', 'do', var]  # Use walk as "hop" proxy
            mod_actions = ['WALK']
            
            test_cases.append({
                'original': {'command': orig_command, 'actions': orig_actions},
                'modified': {'command': mod_command, 'actions': mod_actions},
                'modification_type': 'simple_substitution'
            })
        
        # Test 2: Compositional modification
        # "X means jump. Do X twice." but with X->walk
        for var in self.variables[:2]:
            orig_command = [var, 'means', 'jump', 'do', var, 'twice']
            orig_actions = ['JUMP', 'JUMP']
            
            mod_command = [var, 'means', 'walk', 'do', var, 'twice']
            mod_actions = ['WALK', 'WALK']
            
            test_cases.append({
                'original': {'command': orig_command, 'actions': orig_actions},
                'modified': {'command': mod_command, 'actions': mod_actions},
                'modification_type': 'compositional_substitution'
            })
        
        return test_cases


def test_task_generation():
    """Test the task generator."""
    generator = DereferencingTaskGenerator()
    
    logger.info("Testing simple binding generation...")
    command, actions, bindings = generator.generate_simple_binding()
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Actions: {' '.join(actions)}")
    logger.info(f"Bindings: {bindings}")
    
    logger.info("\nTesting multiple bindings...")
    command, actions, bindings = generator.generate_multiple_bindings()
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Actions: {' '.join(actions)}")
    logger.info(f"Bindings: {bindings}")
    
    logger.info("\nTesting rebinding...")
    command, actions, bindings = generator.generate_rebinding()
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Actions: {' '.join(actions)}")
    logger.info(f"Bindings: {bindings}")
    
    logger.info("\nGenerating small dataset...")
    dataset = generator.generate_dataset(n_samples=100)
    logger.info(f"Train samples: {dataset['train']['commands'].shape[0]}")
    logger.info(f"Val samples: {dataset['val']['commands'].shape[0]}")
    logger.info(f"Test samples: {dataset['test']['commands'].shape[0]}")


if __name__ == "__main__":
    test_task_generation()