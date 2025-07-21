# OOD Illusion Paper - Draft Status

## Overview
This directory contains the draft of our paper "The OOD Illusion in Physics Learning: When Generalization Methods Make Things Worse" which documents our findings that all major OOD methods fail catastrophically on true physics extrapolation.

## Key Results
- Test-Time Adaptation (TTA): 235% worse than baseline
- MAML with adaptation: 62,290% worse than baseline  
- GFlowNet/Ensembles: ~1% improvement (negligible)
- Core insight: Current "OOD" benchmarks test interpolation, not extrapolation

## Draft Structure
The paper is organized into modular sections for easier editing:

1. `introduction_revised.md` - Sets up the problem and key findings
2. `background_draft.md` - Reviews OOD methods and benchmarks
3. `methods_draft.md` - Describes our physics extrapolation setup
4. `results_draft.md` - Presents empirical findings
5. `analysis_draft.md` - Explains why methods fail
6. `taxonomy_draft.md` - Proposes shift categorization
7. `future_directions_draft.md` - Discusses implications
8. `conclusion_draft.md` - Summarizes contributions

The complete paper is in `full_paper_draft.md` (sections need to be merged).

## Writing Principles
- Objective, measured language (no hyperbole)
- Let the data speak (62,290% is shocking enough)
- Acknowledge limitations
- Focus on constructive insights

## Next Steps
1. Review and refine individual sections
2. Merge sections into complete draft
3. Add references (currently placeholder)
4. Create figures (currently described in text)
5. Write detailed appendix
6. Internal review and revision

## Target Venues
- ICLR 2025 (September/October deadline) - most immediate
- ICML 2025 (January deadline)
- NeurIPS 2025 (May deadline)

## Key Messages
1. Methods designed to help OOD generalization can make things dramatically worse
2. This happens because current benchmarks test the wrong kind of generalization
3. True extrapolation (mechanism changes) requires fundamentally different approaches
4. We need new benchmarks and methods for genuine OOD scenarios