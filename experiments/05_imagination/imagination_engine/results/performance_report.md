# ARC Performance Analysis Report

*Generated: 2025-08-22T14:50:03.178298*

*Dataset: 100 tasks*

## Executive Summary

- **Overall Success Rate**: 20.0%
- **Tasks Solved**: 20/100
- **Average Accuracy**: 48.2%
- **Perfect Solutions**: 1
- **Average Time per Task**: 0.05s

## Key Findings

⚠️ **Potential overfitting detected** (performance drop: -8.0%)

### Strongest Task Categories:
- **other**: 40.0% success rate (10/25 tasks)
- **color_mapping**: 38.1% success rate (8/21 tasks)
- **object_duplication**: 25.0% success rate (2/8 tasks)

### Areas Needing Improvement:
- **error**: 0.0% success rate (0/44 tasks)
- **resize**: 0.0% success rate (0/2 tasks)
- **object_duplication**: 25.0% success rate (2/8 tasks)

## Detailed Performance Analysis

### Performance by Task Type

| Task Type | Total | Solved | Success Rate | Avg Accuracy |
|-----------|-------|--------|--------------|--------------|
| color_mapping   |    21 |      8 |        38.1% |        86.2% |
| error           |    44 |      0 |         0.0% |         0.0% |
| object_duplication |     8 |      2 |        25.0% |        85.8% |
| other           |    25 |     10 |        40.0% |        87.7% |
| resize          |     2 |      0 |         0.0% |        65.3% |

### Overfitting Analysis

- **First Half Performance**: 24.0% (12 tasks)
- **Second Half Performance**: 16.0% (8 tasks)
- **Performance Delta**: -8.0%

### Most Effective Primitives

| Rank | Primitive | Uses |
|------|-----------|------|
|    1 | fill_enclosed_1_2(boundary=1, fill=2)              |    8 |
|    2 | fill_enclosed_2_3(boundary=2, fill=3)              |    5 |
|    3 | fill_enclosed_2_4(boundary=2, fill=4)              |    5 |
|    4 | fill_enclosed_2_1(boundary=2, fill=1)              |    3 |
|    5 | fill_enclosed_1_3(boundary=1, fill=3)              |    2 |
|    6 | fill_enclosed_2_8(boundary=2, fill=8)              |    2 |
|    7 | fill_enclosed_2_5(boundary=2, fill=5)              |    2 |
|    8 | swap_colors_1_2(color1=1, color2=2) -> map_2_to... |    1 |
|    9 | fill_enclosed_6_3(boundary=6, fill=3)              |    1 |
|   10 | fill_enclosed_4_6(boundary=4, fill=6)              |    1 |
|   11 | resize_crop_(4, 4)(shape=(4, 4)) -> map_1_to_0(... |    1 |
|   12 | fill_enclosed_3_2(boundary=3, fill=2)              |    1 |
|   13 | swap_colors_1_6(color1=1, color2=6)                |    1 |
|   14 | fill_enclosed_4_2(boundary=4, fill=2)              |    1 |
|   15 | fill_enclosed_5_8(boundary=5, fill=8)              |    1 |

## Failure Pattern Analysis

### Failed Tasks: 80/100

**error** (44 failures):
  Examples: a85d4709, c8cbb738, 8e1813be, 20fb2937, 5c2c9af4

**other** (15 failures):
  Examples: f83cb3f6, f0100645, 46c35fc7, fc4aaf52, fe45cba4

**color_mapping** (13 failures):
  Examples: ff72ca3e, bdad9b1f, d06dbe63, 00dbd492, e760a62e

**object_duplication** (6 failures):
  Examples: 94414823, baf41dbf, ecdecbb3, 4c5c2cf0, f8be4b64

**resize** (2 failures):
  Examples: 94f9d214, 6773b310

## Recommendations
- **Critical**: Overall solve rate below 40% - consider adding more fundamental primitives
- **Focus Area**: error tasks have very low success rate (0.0%) - analyze failed examples
- **Missing Capability**: Resize operations need improvement - add more flexible grid manipulation