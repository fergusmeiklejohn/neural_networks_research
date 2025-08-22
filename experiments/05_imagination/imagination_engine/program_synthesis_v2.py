"""Enhanced Program Synthesis with Extended Primitives.

Integrates the extended primitive library for better ARC task coverage.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import time

from arc_primitives import ARCPrimitives
from arc_primitives_extended import ARCPrimitivesExtended


class Transform:
    """Base class for all transformations."""
    def apply(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def to_string(self) -> str:
        raise NotImplementedError


@dataclass
class Primitive(Transform):
    """A primitive transformation."""
    name: str
    func: Callable
    params: Dict[str, Any] = None
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        try:
            return self.func(grid)
        except:
            return grid
    
    def to_string(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return self.name


@dataclass
class Sequence(Transform):
    """Sequential composition of transformations."""
    transforms: List[Transform]
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid
        for t in self.transforms:
            result = t.apply(result)
        return result
    
    def to_string(self) -> str:
        return " -> ".join(t.to_string() for t in self.transforms)


@dataclass 
class Conditional(Transform):
    """Conditional transformation."""
    condition: Callable
    if_true: Transform
    if_false: Optional[Transform] = None
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        if self.condition(grid):
            return self.if_true.apply(grid)
        elif self.if_false:
            return self.if_false.apply(grid)
        return grid
    
    def to_string(self) -> str:
        s = f"if(condition) then {self.if_true.to_string()}"
        if self.if_false:
            s += f" else {self.if_false.to_string()}"
        return s


class EnhancedProgramSynthesizer:
    """Enhanced synthesizer with extended primitives."""
    
    def __init__(self):
        self.max_program_length = 5
        self.beam_size = 10  # For beam search
        
    def synthesize(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                  max_time: float = 10.0) -> Optional[Transform]:
        """Synthesize a program using beam search."""
        
        start_time = time.time()
        
        # Try primitives first
        primitive = self._beam_search_primitives(examples, max_time/2)
        if primitive:
            return primitive
        
        # Try sequences
        if time.time() - start_time < max_time:
            sequence = self._beam_search_sequences(examples, 
                                                  max_time - (time.time() - start_time))
            if sequence:
                return sequence
        
        return None
    
    def _beam_search_primitives(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                max_time: float) -> Optional[Transform]:
        """Use beam search to find best primitive."""
        
        candidates = self._enumerate_all_primitives(examples)
        beam = []
        
        start_time = time.time()
        
        for candidate in candidates:
            if time.time() - start_time > max_time:
                break
                
            score = self._evaluate(candidate, examples)
            beam.append((score, candidate))
        
        # Sort by score and keep top candidates
        beam.sort(key=lambda x: -x[0])
        beam = beam[:self.beam_size]
        
        if beam and beam[0][0] > 0.9:
            print(f"Found primitive: {beam[0][1].to_string()} (score={beam[0][0]:.2f})")
            return beam[0][1]
        
        return None
    
    def _beam_search_sequences(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                               max_time: float) -> Optional[Transform]:
        """Use beam search to find best sequence."""
        
        # Start with single primitives
        primitives = self._enumerate_all_primitives(examples)
        beam = [(self._evaluate(p, examples), [p]) for p in primitives[:20]]
        beam.sort(key=lambda x: -x[0])
        beam = beam[:self.beam_size]
        
        start_time = time.time()
        
        for length in range(2, self.max_program_length + 1):
            if time.time() - start_time > max_time:
                break
                
            new_beam = []
            
            for score, sequence in beam:
                # Try extending with each primitive type
                extensions = self._get_sequence_extensions(sequence[-1], examples)
                
                for ext in extensions[:5]:  # Limit extensions per sequence
                    new_seq = sequence + [ext]
                    seq_transform = Sequence(new_seq)
                    new_score = self._evaluate(seq_transform, examples)
                    
                    if new_score > score:  # Only keep if improving
                        new_beam.append((new_score, new_seq))
            
            # Merge and prune beam
            beam.extend(new_beam)
            beam.sort(key=lambda x: -x[0])
            beam = beam[:self.beam_size]
            
            # Early stopping if we found perfect solution
            if beam and beam[0][0] >= 1.0:
                print(f"Found perfect sequence: {Sequence(beam[0][1]).to_string()}")
                return Sequence(beam[0][1])
        
        if beam and beam[0][0] > 0.7:
            print(f"Best sequence: {Sequence(beam[0][1]).to_string()} (score={beam[0][0]:.2f})")
            return Sequence(beam[0][1])
        
        return None
    
    def _enumerate_all_primitives(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Transform]:
        """Enumerate all applicable primitives with parameters."""
        
        candidates = []
        
        # Analyze examples
        all_input_colors = set()
        all_output_colors = set()
        components_found = []
        
        for input_grid, output_grid in examples:
            all_input_colors.update(np.unique(input_grid))
            all_output_colors.update(np.unique(output_grid))
            
            # Find components
            comps = ARCPrimitives.find_connected_components(input_grid)
            if comps:
                components_found.extend(comps)
        
        colors = sorted(all_input_colors | all_output_colors)
        
        # === Basic Transformations ===
        
        # Color operations
        for c1 in colors:
            for c2 in colors:
                if c1 != c2 and c1 != 0 and c2 != 0:
                    # Color swapping
                    candidates.append(Primitive(
                        f"swap_colors_{c1}_{c2}",
                        lambda g, a=c1, b=c2: ARCPrimitivesExtended.swap_colors(g, a, b),
                        {"color1": c1, "color2": c2}
                    ))
                    
                    # Fill enclosed regions
                    candidates.append(Primitive(
                        f"fill_enclosed_{c1}_{c2}",
                        lambda g, b=c1, f=c2: ARCPrimitivesExtended.fill_enclosed_regions(g, b, f),
                        {"boundary": c1, "fill": c2}
                    ))
        
        # Grid transformations
        candidates.extend([
            Primitive("rotate_90", lambda g: np.rot90(g, 1)),
            Primitive("rotate_180", lambda g: np.rot90(g, 2)),
            Primitive("rotate_270", lambda g: np.rot90(g, 3)),
            Primitive("flip_horizontal", lambda g: np.flip(g, axis=1)),
            Primitive("flip_vertical", lambda g: np.flip(g, axis=0)),
            Primitive("transpose", lambda g: g.T),
        ])
        
        # === Size-based operations ===
        
        if len(examples) > 0:
            # Check if output size differs from input
            inp_shape = examples[0][0].shape
            out_shape = examples[0][1].shape
            
            if inp_shape != out_shape:
                # Resizing operations
                candidates.extend([
                    Primitive(f"resize_crop_{out_shape}", 
                             lambda g, s=out_shape: ARCPrimitivesExtended.resize_grid(g, s, 'crop'),
                             {"shape": out_shape}),
                    Primitive(f"resize_pad_{out_shape}",
                             lambda g, s=out_shape: ARCPrimitivesExtended.resize_grid(g, s, 'pad'),
                             {"shape": out_shape}),
                    Primitive(f"resize_repeat_{out_shape}",
                             lambda g, s=out_shape: ARCPrimitivesExtended.resize_grid(g, s, 'repeat'),
                             {"shape": out_shape})
                ])
                
                # Scale operations
                if out_shape[0] % inp_shape[0] == 0 and out_shape[1] % inp_shape[1] == 0:
                    scale = out_shape[0] // inp_shape[0]
                    candidates.append(Primitive(f"scale_{scale}x",
                                               lambda g, s=scale: np.repeat(np.repeat(g, s, axis=0), s, axis=1),
                                               {"scale": scale}))
        
        # === Object-based operations ===
        
        if components_found:
            # For the first component found, try various operations
            comp = components_found[0]
            
            # Rotation
            for angle in [90, 180, 270]:
                candidates.append(Primitive(
                    f"rotate_object_{angle}",
                    lambda g, c=comp, a=angle: ARCPrimitivesExtended.rotate_object(g, c, a),
                    {"angle": angle}
                ))
            
            # Mirroring
            candidates.extend([
                Primitive("mirror_horizontal",
                         lambda g, c=comp: ARCPrimitivesExtended.mirror_object(g, c, 'horizontal'),
                         {}),
                Primitive("mirror_vertical", 
                         lambda g, c=comp: ARCPrimitivesExtended.mirror_object(g, c, 'vertical'),
                         {})
            ])
            
            # Scaling
            for scale in [2, 3]:
                candidates.append(Primitive(
                    f"scale_object_{scale}x",
                    lambda g, c=comp, s=scale: ARCPrimitivesExtended.scale_object(g, c, s),
                    {"scale": scale}
                ))
            
            # Duplication patterns
            if len(examples) > 0:
                h, w = examples[0][1].shape
                
                # Grid arrangements
                for rows in [2, 3]:
                    for cols in [2, 3]:
                        if rows * comp.bounding_box[2] <= h and cols * comp.bounding_box[3] <= w:
                            candidates.append(Primitive(
                                f"grid_{rows}x{cols}",
                                lambda g, c=comp, r=rows, cl=cols: 
                                    ARCPrimitivesExtended.create_grid_of_objects(g, c, r, cl),
                                {"rows": rows, "cols": cols}
                            ))
        
        # === Pattern continuation ===
        
        candidates.extend([
            Primitive("continue_right", 
                     lambda g: ARCPrimitivesExtended.continue_pattern(g, 'right', 1)),
            Primitive("continue_down",
                     lambda g: ARCPrimitivesExtended.continue_pattern(g, 'down', 1)),
            Primitive("continue_radial",
                     lambda g: ARCPrimitivesExtended.continue_pattern(g, 'radial', 1))
        ])
        
        # === Sorting operations ===
        
        candidates.extend([
            Primitive("sort_by_size_h",
                     lambda g: ARCPrimitivesExtended.sort_objects_by_size(g, 'horizontal')),
            Primitive("sort_by_size_v", 
                     lambda g: ARCPrimitivesExtended.sort_objects_by_size(g, 'vertical')),
            Primitive("sort_colors_freq",
                     lambda g: ARCPrimitivesExtended.sort_colors_by_frequency(g))
        ])
        
        return candidates
    
    def _get_sequence_extensions(self, last_transform: Transform, 
                                examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Transform]:
        """Get potential extensions for a sequence based on the last transform."""
        
        extensions = []
        
        # Get intermediate result after last transform
        intermediate = last_transform.apply(examples[0][0])
        
        # Analyze what changed
        input_colors = set(np.unique(examples[0][0]))
        intermediate_colors = set(np.unique(intermediate))
        target_colors = set(np.unique(examples[0][1]))
        
        # If colors are missing, try color operations
        if intermediate_colors != target_colors:
            for c1 in intermediate_colors:
                for c2 in target_colors:
                    if c1 != c2 and c1 != 0:
                        extensions.append(Primitive(
                            f"map_{c1}_to_{c2}",
                            lambda g, a=c1, b=c2: ARCPrimitivesExtended.map_colors(g, {a: b}),
                            {"from": c1, "to": c2}
                        ))
        
        # If size is wrong, try resizing
        if intermediate.shape != examples[0][1].shape:
            target_shape = examples[0][1].shape
            extensions.extend([
                Primitive(f"resize_to_{target_shape}",
                         lambda g, s=target_shape: ARCPrimitivesExtended.resize_grid(g, s, 'crop')),
                Primitive(f"pad_to_{target_shape}",
                         lambda g, s=target_shape: ARCPrimitivesExtended.resize_grid(g, s, 'pad'))
            ])
        
        # Try pattern operations
        extensions.extend([
            Primitive("flip_h", lambda g: np.flip(g, axis=1)),
            Primitive("flip_v", lambda g: np.flip(g, axis=0)),
            Primitive("rotate_90", lambda g: np.rot90(g, 1))
        ])
        
        return extensions
    
    def _evaluate(self, transform: Transform, examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate a transform on examples."""
        
        if not examples:
            return 0.0
        
        total_score = 0.0
        
        for input_grid, expected_output in examples:
            try:
                predicted = transform.apply(input_grid)
                
                # Check shape match
                if predicted.shape != expected_output.shape:
                    continue
                
                # Calculate accuracy
                accuracy = np.mean(predicted == expected_output)
                total_score += accuracy
                
            except Exception:
                continue
        
        return total_score / len(examples)