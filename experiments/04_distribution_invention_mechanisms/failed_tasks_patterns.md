# Failed Task Analysis Summary

Total tasks evaluated: 30

Tasks solved: 2 (6.7%)
Tasks failed: 28 (93.3%)

## Missing DSL Primitives

Based on analysis of failed tasks, we need:


### Conditional Fill
- Found in 306 failed tasks (1092.9%)
- Example tasks: ae3edfdc, ae3edfdc, ae3edfdc, ae3edfdc, ae3edfdc

### Sorting By Size
- Found in 23 failed tasks (82.1%)
- Example tasks: ae3edfdc, d406998b, 8e5a5113, 508bd3b6, db93a21d

### Counting
- Found in 22 failed tasks (78.6%)
- Example tasks: d406998b, 8403a5d5, 8e5a5113, 508bd3b6, db93a21d

### Line Drawing
- Found in 16 failed tasks (57.1%)
- Example tasks: 8403a5d5, 8403a5d5, db93a21d, db93a21d, 36d67576

### Pattern Propagation
- Found in 5 failed tasks (17.9%)
- Example tasks: 8403a5d5, 74dd1130, 25ff71a9, 29c11459, 88a62173

### Grid Partition
- Found in 3 failed tasks (10.7%)
- Example tasks: 53b68214, c59eb873, 67e8384a

## Recommendations

Priority primitives to implement:

1. **Line Drawing**: DrawLine, ConnectPoints with straight lines
2. **Counting**: CountObjects, CountByColor, CountBySize
3. **Grid Partition**: PartitionGrid, ExtractQuadrant, MergeRegions
4. **Conditional Fill**: FillIfNeighbor, PropagateColor, FloodFillConditional
5. **Edge Detection**: ExtractBoundaries, TraceBorder, GetPerimeter
6. **Pattern Propagation**: ExtendPattern, RepeatUntilEdge, FillWithPattern
7. **Sorting**: SortBySize, SortByPosition, ArrangeInOrder
