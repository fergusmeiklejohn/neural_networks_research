# Response to Reviewers - Summary

Thank you for your thoughtful feedback. We have substantially revised the paper based on your suggestions. Here is a summary of the major changes:

## 1. Additional Empirical Work

**Reviewer concern**: "Would have liked another task beyond balls and gravity"

**Our response**: We added a complete second physics system - pendulum with time-varying length:
- Implemented comprehensive experiments (Section 4.2)
- Tested physics-aware TTA variants using energy and Hamiltonian consistency
- Added PeTTA-inspired collapse detection
- Results show 12-18x performance degradation, confirming generality of findings

## 2. Contemporary Literature Integration

**Reviewer concern**: Missing recent advances in TTA (PeTTA, TAIP, TTAB)

**Our response**: Added new Section 2.3 "Recent Advances in Stabilized Test-Time Adaptation":
- Discusses PeTTA's collapse detection mechanisms
- Explains TAIP's physics-aware adaptation success
- References TTAB's comprehensive benchmark findings
- Positions our work as complementary, identifying where these methods reach limits

## 3. Statistical Rigor

**Reviewer concern**: Need confidence intervals, p-values, statistical significance

**Our response**: All results now include:
- 95% confidence intervals (bootstrap with 1000 samples)
- Two-sided t-tests with n=5 seeds
- p-values for all comparisons
- Clear indication of statistical significance

## 4. Tempered Language and Claims

**Reviewer concern**: Overly dramatic language, unsupported universal claims

**Our response**: Systematically revised language throughout:
- Removed hyperbole ("catastrophic", "shocking")
- Added scope qualifiers ("in physics tasks with mechanism shifts")
- Let numbers speak for themselves
- Positioned findings as complementary to recent advances

## 5. Experimental Breadth

**Reviewer concern**: Only one type of adaptation tested

**Our response**: Implemented multiple adaptation variants:
- Standard prediction consistency TTA
- Energy conservation TTA
- Hamiltonian consistency TTA
- PeTTA-inspired collapse detection
- All show degradation (0.06% improvement with collapse detection)

## 6. Clarified Positioning

**Reviewer concern**: Seems to contradict recent TTA successes

**Our response**: Added Section 5.4 explaining relationship:
- PeTTA prevents collapse; we show stability isn't enough for mechanism shifts
- TAIP uses physics constraints; we show these fail when physics changes
- Identified three scenarios: parameter adaptation, stability preservation, mechanism learning
- Our work identifies the frontier where new approaches are needed

## Key Finding Remains

Despite all improvements and additional experiments, the core finding holds: current self-supervised adaptation methods degrade performance on mechanism shifts where new computational operations are required. This is now demonstrated across two physics systems with multiple adaptation variants and proper statistical analysis.

## Paper Structure

The revised paper includes:
- Complete rewrite of introduction with measured tone
- New background section on recent TTA advances
- Comprehensive methods covering both physics systems
- Full statistical analysis in results
- New discussion section on contemporary methods
- Taxonomy refined based on feedback
- Conclusion properly scoped to our evidence

All reviewer concerns have been addressed while maintaining the paper's core contribution.