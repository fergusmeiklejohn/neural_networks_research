# Language Softening Revisions

## Key Changes to Address Reviewer Concern

The reviewer noted that phrases like "universal failure" and "fundamental limitations" overreach given our single domain study. Here are the specific revisions:

### 1. Abstract
**Original**: "revealing universal failure of current methods"
**Revised**: "revealing systematic failure of tested methods on this benchmark"

### 2. Results Section 4.3.2
**Original**: "Universal Model Failure"
**Revised**: "Systematic Model Failure on Time-Varying Gravity"

**Original**: "All models showed catastrophic failure"
**Revised**: "All tested models showed substantial performance degradation"

### 3. Results Section 4.4
**Original**: "True extrapolation remains unsolved"
**Revised**: "True extrapolation remains challenging for current approaches in physics learning"

### 4. Discussion Section 5.1.1
**Original**: "This phenomenon explains why models can achieve low error on 'OOD' test sets while failing catastrophically on genuinely novel physics."
**Revised**: "This phenomenon may explain why models can achieve low error on 'OOD' test sets while showing substantial degradation on structurally different physics scenarios."

### 5. Discussion Section 5.2.1
**Original**: "When physics constraints are rigidly encoded (e.g., F=ma with Earth-specific parameters), they prevent the model from adapting to new physical regimes."
**Revised**: "When physics constraints are rigidly encoded (e.g., F=ma with Earth-specific parameters), they may limit the model's ability to adapt to new physical regimes, though recent work suggests flexible physics-inspired designs can help (Kim et al., 2025)."

### 6. Discussion Section 5.3.3
**Original**: "suggest that pure neural methods may have fundamental limitations"
**Revised**: "suggest that pure neural methods face significant challenges"

### 7. Discussion Section 5.4.2
**Original**: "creates illusions about their extrapolation capabilities"
**Revised**: "may create overconfidence in their extrapolation capabilities"

### 8. Conclusion
**Original**: "When tested on genuinely out-of-distribution scenarios involving time-varying gravity, all models fail catastrophically."
**Revised**: "When tested on our time-varying gravity benchmark, all evaluated models showed substantial performance degradation, consistent with recent theoretical understanding of spectral shifts in PINNs (Fesser et al., 2023)."

**Original**: "fundamental limitations in current purely neural approaches"
**Revised**: "significant challenges facing current neural approaches"

### 9. Throughout the Paper
Replace all instances of:
- "universal" → "systematic" or "consistent"
- "catastrophic failure" → "substantial degradation" or "significant performance drop"
- "fundamental limitations" → "current limitations" or "significant challenges"
- "proves" → "suggests" or "indicates"
- "all models" → "all tested models" or "the evaluated models"

### 10. Add Qualifiers Where Appropriate
- "in our experiments"
- "for the physics domain studied"
- "within the scope of our analysis"
- "based on our 2D dynamics benchmark"

## Example Paragraph Revision

**Original**:
"This provides definitive evidence that structural changes in physics create genuinely out-of-distribution scenarios that current methods cannot handle through interpolation."

**Revised**:
"This provides strong evidence that structural changes in physics, such as time-varying parameters, create genuinely out-of-distribution scenarios that the tested methods cannot handle through interpolation in our 2D dynamics domain. This aligns with recent theoretical work on spectral shifts causing PINN extrapolation failures (Fesser et al., 2023)."

## Rationale

These changes:
1. Acknowledge the limited scope of our study (2D ball dynamics)
2. Avoid claims about all possible models or domains
3. Replace absolute statements with evidence-based observations
4. Connect findings to supporting literature
5. Maintain the strength of our evidence while being precise about scope
