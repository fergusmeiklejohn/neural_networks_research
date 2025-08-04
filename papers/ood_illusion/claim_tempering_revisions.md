# Claim Tempering Revisions

## Overview
The reviewer correctly notes that our claims sound too universal. We need to scope them specifically to "physics tasks with mechanism shifts" while maintaining impact.

## Abstract Revisions

### Original Claims (Too Broad)
> "Our results suggest that achieving genuine OOD generalization may require fundamentally different approaches than those currently popular in the field."

### Revised Claims (Properly Scoped)
> "Our results suggest that achieving OOD generalization on physics tasks with mechanism shifts—where the underlying generative process changes—may require approaches beyond current self-supervised adaptation methods."

### Original Finding Statement
> "We find that current state-of-the-art OOD methods fail catastrophically"

### Revised Finding Statement
> "We find that on physics prediction tasks with time-varying mechanisms, current adaptation methods show significant performance degradation"

## Introduction Revisions

### Original Opening
> "Machine learning models often fail when deployed outside their training distribution. This observation has motivated extensive research into out-of-distribution (OOD) generalization..."

### Revised Opening
> "Machine learning models for physics prediction face a particular challenge: when the underlying physical mechanisms change over time, performance can degrade catastrophically. While recent advances in OOD generalization show promise for many distribution shifts, we investigate a specific class—mechanism changes—that poses unique challenges."

### Original Claim
> "This suggests that genuine OOD generalization remains an open problem"

### Revised Claim
> "This suggests that OOD generalization for mechanism shifts in physics remains an open problem, distinct from the statistical distribution shifts addressed by current methods"

## Methods Section Revisions

### Original
> "We evaluate whether current OOD methods can handle true extrapolation"

### Revised
> "We evaluate whether current OOD methods can handle mechanism shifts—a specific form of distribution shift where the data-generating process changes"

## Results Section Revisions

### Original
> "No tested method improves over the baseline"

### Revised
> "On tasks with mechanism shifts, no tested method showed statistically significant improvement over the baseline (p > 0.05, n=5 seeds)"

## Discussion Section Revisions

### Original
> "Current OOD methods are fundamentally limited"

### Revised
> "Current OOD methods show systematic limitations when facing mechanism shifts in physics, though recent work like PeTTA [cite] and TAIP [cite] demonstrates progress on other types of distribution shift"

## Taxonomy Section Revisions

### Original
> "This taxonomy explains why current methods fail"

### Revised
> "This taxonomy helps explain why current self-supervised adaptation methods show reduced performance on Level 3 (mechanism) shifts while succeeding on Level 1-2 shifts"

## Conclusion Revisions

### Original Opening
> "We have shown that current OOD methods fail on true extrapolation"

### Revised Opening
> "We have shown that on physics tasks involving mechanism shifts—where the data-generating equations change—current self-supervised adaptation methods show systematic performance degradation"

### Original Implications
> "These results have broad implications for machine learning"

### Revised Implications
> "These results have implications for physics-based machine learning and other domains where mechanism shifts occur, such as climate modeling with tipping points or financial markets with regime changes"

## Key Phrases to Add Throughout

When making broad claims, add qualifiers:
- "in physics tasks with mechanism shifts"
- "for self-supervised adaptation methods"
- "when the generative process changes"
- "in our experimental settings"
- "for the class of problems we study"

## Specific Scoping for Each Section

1. **Abstract**: Add "physics tasks with mechanism shifts" to main claim
2. **Introduction**: Open with physics-specific challenge, not general OOD
3. **Background**: Acknowledge recent progress while identifying gap
4. **Methods**: Define "mechanism shift" precisely
5. **Results**: Add statistical significance and confidence intervals
6. **Analysis**: Explain why physics mechanism shifts are special
7. **Taxonomy**: Position as complementary to existing frameworks
8. **Future Work**: Suggest this is one important direction among many
9. **Conclusion**: Scope implications to relevant domains

## Example of Properly Scoped Claim

### Too Broad
"Test-time adaptation fails on OOD data"

### Properly Scoped
"Test-time adaptation using self-supervised consistency losses shows performance degradation on physics prediction tasks where the underlying mechanism (e.g., gravitational dynamics) changes during deployment"

This maintains scientific impact while being precise about scope.
