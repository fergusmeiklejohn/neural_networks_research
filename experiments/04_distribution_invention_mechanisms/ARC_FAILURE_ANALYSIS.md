# ARC-AGI Failure Analysis Report

Analyzed 50 tasks

## Failure Categories

- **missing_arithmetic**: 26 tasks (52.0%)
- **missing_conditional_logic**: 22 tasks (44.0%)
- **missing_spatial_reasoning**: 2 tasks (4.0%)

## Top Missing Capabilities

- `spatial:spiral`: 42 occurrences
- `spatial:repeating`: 36 occurrences
- `conditional:if_square_then_transform`: 29 occurrences
- `counting:duplication`: 22 occurrences
- `structural:objects_split`: 22 occurrences
- `counting:reduction`: 20 occurrences
- `structural:objects_merged`: 20 occurrences
- `arithmetic:add_0_to_colors`: 17 occurrences
- `spatial:symmetric`: 15 occurrences
- `conditional:if_large_then_fill`: 12 occurrences
- `arithmetic:count_encoding`: 10 occurrences
- `spatial:diagonal`: 6 occurrences
- `conditional:if_color_2_then_0`: 5 occurrences
- `spatial:border`: 4 occurrences
- `structural:objects_rearranged`: 4 occurrences
- `conditional:if_color_2_then_6`: 2 occurrences
- `conditional:if_color_3_then_0`: 2 occurrences
- `conditional:if_color_8_then_0`: 2 occurrences
- `conditional:if_color_1_then_0`: 2 occurrences
- `conditional:if_color_4_then_0`: 2 occurrences

## Capability Categories

### Arithmetic Operations

- add_0_to_colors: 17 tasks
- count_encoding: 10 tasks

### Conditional Operations

- if_square_then_transform: 29 tasks
- if_large_then_fill: 12 tasks
- if_color_2_then_0: 5 tasks
- if_color_2_then_6: 2 tasks
- if_color_3_then_0: 2 tasks
- if_color_8_then_0: 2 tasks
- if_color_1_then_0: 2 tasks
- if_color_4_then_0: 2 tasks
- if_color_6_then_0: 2 tasks
- if_color_9_then_0: 2 tasks
- if_color_1_then_5: 1 tasks
- if_color_3_then_4: 1 tasks
- if_color_4_then_3: 1 tasks
- if_color_5_then_1: 1 tasks
- if_color_6_then_2: 1 tasks
- if_color_8_then_9: 1 tasks
- if_color_9_then_8: 1 tasks
- if_color_2_then_3: 1 tasks
- if_color_2_then_4: 1 tasks
- if_color_2_then_7: 1 tasks

### Spatial Operations

- spiral: 42 tasks
- repeating: 36 tasks
- symmetric: 15 tasks
- diagonal: 6 tasks
- border: 4 tasks

### Counting Operations

- duplication: 22 tasks
- reduction: 20 tasks

### Structural Operations

- objects_split: 22 tasks
- objects_merged: 20 tasks
- objects_rearranged: 4 tasks

## Recommendations for DSL Enhancement

Based on this analysis, we should add:

### 1. Arithmetic Primitives
- `AddConstant(value)`: Add constant to all non-zero colors
- `CountObjects()`: Count objects and encode as color
- `MultiplyColors(factor)`: Scale color values

### 2. Conditional Logic
- `IfSize(threshold, then_op, else_op)`: Size-based conditions
- `IfColor(color, then_op)`: Color-based conditions
- `IfShape(shape_test, then_op)`: Shape-based conditions

### 3. Spatial Patterns
- `DrawDiagonal(color)`: Draw diagonal lines
- `DrawBorder(color, thickness)`: Draw borders
- `FillSpiral(start_color)`: Fill in spiral pattern
- `RepeatPattern(pattern, times)`: Repeat spatial pattern

### 4. Counting and Indexing
- `EnumerateObjects()`: Number objects sequentially
- `DuplicateNTimes(n)`: Controlled duplication
- `SelectNth(n)`: Select nth object

### 5. Structural Operations
- `MergeAdjacent()`: Merge touching objects
- `SplitByColor()`: Split multi-color objects
- `ConnectObjects(method)`: Connect objects with lines


## Task-Specific Insights

### Task 007bbfb7
- **Failure**: missing_conditional_logic
- **Missing**: conditional:if_large_then_fill, conditional:if_square_then_transform, counting:duplication, spatial:repeating, spatial:spiral, spatial:symmetric, structural:objects_split
- **Spatial**: spiral,symmetric
- **Conditional**: if_large_then_fill,if_square_then_transform

### Task 00d62c1b
- **Failure**: missing_arithmetic
- **Missing**: arithmetic:count_encoding, conditional:if_square_then_transform, counting:duplication, spatial:repeating, spatial:spiral, spatial:symmetric, structural:objects_split
- **Arithmetic**: count_encoding
- **Spatial**: symmetric

### Task 017c7c7b
- **Failure**: missing_conditional_logic
- **Missing**: conditional:if_square_then_transform, counting:duplication, spatial:repeating, spatial:spiral, spatial:symmetric, structural:objects_split
- **Spatial**: spiral,symmetric

### Task 025d127b
- **Failure**: missing_conditional_logic
- **Missing**: conditional:if_square_then_transform, counting:reduction, spatial:repeating, spatial:spiral, structural:objects_merged
- **Spatial**: repeating
- **Conditional**: if_square_then_transform,if_square_then_transform,if_square_then_transform,if_square_then_transform

### Task 045e512c
- **Failure**: missing_arithmetic
- **Missing**: arithmetic:add_0_to_colors, counting:duplication, spatial:repeating, spatial:spiral, structural:objects_split
- **Arithmetic**: add_0_to_colors
- **Spatial**: spiral,repeating

### Task 0520fde7
- **Failure**: missing_conditional_logic
- **Missing**: conditional:if_square_then_transform, counting:reduction, spatial:border, spatial:spiral, spatial:symmetric, structural:objects_merged
- **Spatial**: border,symmetric
- **Conditional**: if_square_then_transform,if_square_then_transform,if_square_then_transform

### Task 05269061
- **Failure**: missing_spatial_reasoning
- **Missing**: counting:duplication, spatial:diagonal, spatial:spiral, structural:objects_split
- **Spatial**: diagonal,spiral

### Task 05f2a901
- **Failure**: missing_arithmetic
- **Missing**: arithmetic:add_0_to_colors, conditional:if_color_2_then_0, conditional:if_large_then_fill, spatial:repeating, spatial:spiral, structural:objects_rearranged
- **Arithmetic**: add_0_to_colors
- **Spatial**: spiral,repeating
- **Conditional**: if_large_then_fill,if_color_2_then_0

### Task 06df4c85
- **Failure**: missing_arithmetic
- **Missing**: arithmetic:add_0_to_colors, conditional:if_square_then_transform, counting:duplication, spatial:repeating, spatial:spiral, structural:objects_split
- **Arithmetic**: add_0_to_colors
- **Spatial**: spiral,repeating
- **Conditional**: if_square_then_transform

### Task 08ed6ac7
- **Failure**: missing_arithmetic
- **Missing**: arithmetic:count_encoding, conditional:if_large_then_fill, spatial:repeating, spatial:spiral
- **Arithmetic**: count_encoding
- **Spatial**: spiral,repeating
- **Conditional**: if_large_then_fill,if_large_then_fill,if_large_then_fill
