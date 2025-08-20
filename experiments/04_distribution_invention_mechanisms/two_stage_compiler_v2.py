#!/usr/bin/env python3
"""Two-Stage Compiler V2 with proper temporal handling.

This version correctly handles rebinding and temporal execution order,
demonstrating how explicit state tracking enables distribution invention.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalBinding:
    """Binding with temporal scope."""

    variable: str
    action: str
    scope_start: int
    scope_end: Optional[int] = None


@dataclass
class ExecutionSegment:
    """Single execution segment with position info."""

    start_pos: int
    end_pos: int
    parse_tree: Optional["ParseNode"] = None


@dataclass
class ParseNode:
    """Simple parse node for execution."""

    node_type: str  # 'var', 'op'
    value: str
    children: List["ParseNode"] = None
    modifier: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class ImprovedBindingExtractor:
    """Improved extractor that handles temporal execution correctly."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}

        # Key tokens
        self.do_token = vocab.get("do", -1)
        self.means_token = vocab.get("means", -1)
        self.and_token = vocab.get("and", -1)
        self.or_token = vocab.get("or", -1)
        self.then_token = vocab.get("then", -1)
        self.twice_token = vocab.get("twice", -1)
        self.thrice_token = vocab.get("thrice", -1)

        # Variables and actions
        self.var_tokens = {
            "X": vocab.get("X", -1),
            "Y": vocab.get("Y", -1),
            "Z": vocab.get("Z", -1),
            "W": vocab.get("W", -1),
        }
        self.action_map = {
            vocab.get("jump", -1): "JUMP",
            vocab.get("walk", -1): "WALK",
            vocab.get("run", -1): "RUN",
            vocab.get("turn", -1): "TURN",
        }

    def extract(
        self, tokens: mx.array
    ) -> Tuple[List[TemporalBinding], List[ExecutionSegment]]:
        """Extract temporal bindings and execution segments."""
        token_list = self._to_list(tokens)

        # Find all binding and execution positions
        bindings = []
        segments = []
        i = 0

        while i < len(token_list):
            # Check for binding pattern: "var means action"
            if i + 2 < len(token_list) and token_list[i + 1] == self.means_token:
                var_token = token_list[i]
                action_token = token_list[i + 2]

                var_name = self._get_var_name(var_token)
                action_name = self.action_map.get(action_token)

                if var_name and action_name:
                    # Create temporal binding
                    binding = TemporalBinding(
                        variable=var_name, action=action_name, scope_start=i
                    )

                    # Update previous binding's scope if same variable
                    for prev_binding in bindings:
                        if (
                            prev_binding.variable == var_name
                            and prev_binding.scope_end is None
                        ):
                            prev_binding.scope_end = i

                    bindings.append(binding)
                    i += 3

                    # Check for immediate "do" after binding
                    if i < len(token_list) and token_list[i] == self.do_token:
                        seg_start = i + 1
                        seg_end, seg_tree = self._parse_execution(token_list, seg_start)
                        segments.append(ExecutionSegment(i, seg_end, seg_tree))
                        i = seg_end
                else:
                    i += 1

            elif token_list[i] == self.do_token:
                # Standalone "do" segment
                seg_start = i + 1
                seg_end, seg_tree = self._parse_execution(token_list, seg_start)
                segments.append(ExecutionSegment(i, seg_end, seg_tree))
                i = seg_end

            else:
                i += 1

        return bindings, segments

    def _to_list(self, tokens: mx.array) -> List[int]:
        """Convert to list."""
        if len(tokens.shape) > 1:
            tokens = tokens[0]
        return [int(t) for t in tokens]

    def _get_var_name(self, token: int) -> Optional[str]:
        """Get variable name from token."""
        for name, tok in self.var_tokens.items():
            if tok == token:
                return name
        return None

    def _parse_execution(
        self, tokens: List[int], start: int
    ) -> Tuple[int, Optional[ParseNode]]:
        """Parse execution segment and return end position and tree."""
        # Find segment end
        end = start
        while end < len(tokens):
            # Stop at next "do" or "means"
            if tokens[end] == self.do_token:
                break
            if end + 1 < len(tokens) and tokens[end + 1] == self.means_token:
                break
            end += 1

        if end <= start:
            return end, None

        # Parse the segment
        tree = self._parse_segment(tokens, start, end)
        return end, tree

    def _parse_segment(
        self, tokens: List[int], start: int, end: int
    ) -> Optional[ParseNode]:
        """Parse a segment into tree structure."""
        # Look for operators
        for i in range(start, end):
            if tokens[i] == self.and_token:
                left = self._parse_segment(tokens, start, i)
                right = self._parse_segment(tokens, i + 1, end)
                if left and right:
                    return ParseNode("op", "AND", [left, right])

            elif tokens[i] == self.or_token:
                left = self._parse_segment(tokens, start, i)
                right = self._parse_segment(tokens, i + 1, end)
                if left and right:
                    return ParseNode("op", "OR", [left, right])

        # No operators - look for variable and modifier
        var_name = None
        modifier = None

        for i in range(start, end):
            v = self._get_var_name(tokens[i])
            if v:
                var_name = v
            elif tokens[i] == self.twice_token:
                modifier = "twice"
            elif tokens[i] == self.thrice_token:
                modifier = "thrice"

        if var_name:
            return ParseNode("var", var_name, modifier=modifier)

        return None


class SimplifiedNeuralExecutor(nn.Module):
    """Simplified neural executor for demonstration."""

    def __init__(self, vocab_size: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def __call__(
        self,
        tokens: mx.array,
        bindings: List[TemporalBinding],
        segments: List[ExecutionSegment],
    ) -> mx.array:
        """Execute based on bindings and segments."""
        outputs = []

        for segment in segments:
            # Get bindings active at this position
            active_bindings = {}
            for binding in bindings:
                if binding.scope_start <= segment.start_pos and (
                    binding.scope_end is None or segment.start_pos < binding.scope_end
                ):
                    active_bindings[binding.variable] = binding.action

            # Execute segment
            if segment.parse_tree:
                self._execute_tree(segment.parse_tree, active_bindings, outputs)

        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((0, self.num_actions))

    def _execute_tree(
        self, node: ParseNode, bindings: Dict[str, str], outputs: List[mx.array]
    ):
        """Execute parse tree."""
        if node.node_type == "var":
            if node.value in bindings:
                action = bindings[node.value]
                action_idx = ["JUMP", "WALK", "RUN", "TURN"].index(action)

                # Create one-hot vector
                vec = mx.zeros(self.num_actions)
                vec = mx.where(mx.arange(self.num_actions) == action_idx, 1.0, vec)

                # Apply modifier
                if node.modifier == "twice":
                    outputs.extend([vec, vec])
                elif node.modifier == "thrice":
                    outputs.extend([vec, vec, vec])
                else:
                    outputs.append(vec)

        elif node.node_type == "op":
            if node.value == "AND":
                self._execute_tree(node.children[0], bindings, outputs)
                self._execute_tree(node.children[1], bindings, outputs)
            elif node.value == "OR":
                self._execute_tree(node.children[0], bindings, outputs)


class TwoStageCompilerV2(nn.Module):
    """Improved Two-Stage Compiler with temporal handling."""

    def __init__(self, vocab_size: int, num_actions: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.extractor = None
        self.executor = SimplifiedNeuralExecutor(vocab_size, num_actions)

    def set_vocab(self, vocab: Dict[str, int]):
        """Set vocabulary."""
        self.extractor = ImprovedBindingExtractor(vocab)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Forward pass."""
        bindings, segments = self.extractor.extract(tokens)
        return self.executor(tokens, bindings, segments)

    def analyze(self, tokens: mx.array) -> Dict:
        """Analyze execution."""
        bindings, segments = self.extractor.extract(tokens)
        outputs = self.executor(tokens, bindings, segments)

        # Convert outputs to actions
        actions = []
        if outputs.shape[0] > 0:
            indices = mx.argmax(outputs, axis=-1)
            action_names = ["JUMP", "WALK", "RUN", "TURN"]
            for idx in indices:
                actions.append(action_names[int(idx)])

        # Format bindings
        binding_info = []
        for b in bindings:
            end = b.scope_end if b.scope_end else "∞"
            binding_info.append(f"{b.variable}->{b.action} [{b.scope_start}:{end}]")

        # Format segments
        segment_info = []
        for i, seg in enumerate(segments):
            active = {}
            for b in bindings:
                if b.scope_start <= seg.start_pos and (
                    b.scope_end is None or seg.start_pos < b.scope_end
                ):
                    active[b.variable] = b.action
            segment_info.append(
                {
                    "position": f"[{seg.start_pos}:{seg.end_pos}]",
                    "tree": self._tree_to_str(seg.parse_tree),
                    "active_bindings": active,
                }
            )

        return {"bindings": binding_info, "segments": segment_info, "actions": actions}

    def _tree_to_str(self, node: Optional[ParseNode]) -> str:
        """Convert tree to string."""
        if not node:
            return "None"
        if node.node_type == "var":
            mod = f" {node.modifier}" if node.modifier else ""
            return f"{node.value}{mod}"
        else:
            left = self._tree_to_str(node.children[0])
            right = self._tree_to_str(node.children[1])
            return f"({left} {node.value} {right})"


def test_improved_compiler():
    """Test the improved compiler."""
    print("=== Testing Improved Two-Stage Compiler ===\n")

    VOCAB = {
        "PAD": 0,
        "do": 1,
        "means": 2,
        "and": 3,
        "or": 4,
        "then": 5,
        "twice": 6,
        "thrice": 7,
        "while": 8,
        "true": 9,
        "X": 10,
        "Y": 11,
        "Z": 12,
        "W": 13,
        "jump": 14,
        "walk": 15,
        "run": 16,
        "turn": 17,
    }

    compiler = TwoStageCompilerV2(len(VOCAB), 4)
    compiler.set_vocab(VOCAB)

    test_cases = [
        ("X means jump do X", ["JUMP"]),
        ("X means jump Y means walk do X and Y", ["JUMP", "WALK"]),
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
        ("X means jump do X then Y means walk do Y and X", ["JUMP", "WALK", "JUMP"]),
    ]

    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")

        # Tokenize
        tokens = [VOCAB.get(w, 0) for w in command.split()]
        tokens_mx = mx.array([tokens])

        # Analyze
        analysis = compiler.analyze(tokens_mx)

        print(f"Bindings: {analysis['bindings']}")
        print("Segments:")
        for seg in analysis["segments"]:
            print(f"  {seg['position']}: {seg['tree']} with {seg['active_bindings']}")
        print(f"Got: {analysis['actions']}")

        if analysis["actions"] == expected:
            print("✅ PASS")
        else:
            print("❌ FAIL")


if __name__ == "__main__":
    test_improved_compiler()
