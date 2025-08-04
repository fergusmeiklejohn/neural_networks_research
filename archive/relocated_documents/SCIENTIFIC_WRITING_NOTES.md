# Scientific Writing Style Guide

Based on the revised introduction, here are key principles for scientific paper writing:

## 1. Tone and Language

### Use Measured Language
- "discovered something interesting" instead of "discovered something shocking"
- "amplify errors by a considerable margin" instead of "catastrophically amplify errors"
- "they fail" instead of "they explode"
- "troubling paradox" instead of "adaptation paradox" (more neutral)

### Avoid Hyperbole
- Remove words like: shocking, catastrophic, explosion, revolutionary, groundbreaking
- Let numbers speak: "235% worse" is striking without additional emphasis
- Use factual descriptors: "considerable", "substantial", "significant"

### State Facts, Not Emotions
- "The method made predictions 235% worse" (fact)
- NOT: "We were shocked to find..." (emotion)

## 2. Claims and Assertions

### Make Defensible Claims
- "what we call" instead of "what the ML community calls" (avoids speaking for others)
- "in most cases" instead of "always" (acknowledges exceptions)
- "suggests we need" instead of "proves we must" (appropriate uncertainty)

### Acknowledge Limitations Implicitly
- "We conducted ablation studies" (shows thoroughness without claiming exhaustiveness)
- "The problem is that" instead of "The fundamental problem" (less absolute)

### Use Active but Careful Voice
- "Our results reveal" (active, taking ownership)
- "This happens because" (explaining mechanism)
- "Our findings challenge an assumption" (not "prove wrong")

## 3. Structure and Flow

### Clear Topic Sentences
- Each paragraph starts with its main point
- Supporting sentences provide evidence
- No buried leads

### Logical Progression
- Start with empirical observation (235% worse)
- Move to broader insight (interpolation vs extrapolation)
- Then to specific analysis (why it happens)
- End with implications (need to rethink)

### Concrete Examples
- Photo→painting vs constant→varying gravity
- Makes abstract concepts tangible

## 4. Technical Precision

### Define Terms Clearly
- "statistical OOD" vs "representational OOD"
- Explain what each means with examples

### Quantify When Possible
- "235% to 62,290% worse" not "much worse"
- Specific numbers add credibility

### Explain Mechanisms
- Don't just say it fails, explain HOW it fails
- "converged to predicting constant values"

## 5. Academic Positioning

### Respectful to Prior Work
- "Current OOD methods excel at the former"
- Acknowledge what works before critiquing

### Constructive Criticism
- Not "methods are wrong" but "methods optimize for the wrong thing"
- Focus on mismatch, not failure

### Forward-Looking
- "suggests we need to radically rethink"
- Offer path forward, not just criticism

## 6. Key Phrases to Use

### For Observations
- "We observed"
- "Our experiments show"
- "The results indicate"

### For Interpretations
- "This suggests"
- "appears to"
- "may indicate"

### For Stronger Claims
- "We demonstrate"
- "The evidence shows"
- "Our analysis reveals"

## 7. What to Avoid

### Inflammatory Language
- "illusion" → "what we call"
- "catastrophically" → "considerably" or just use numbers
- "shocking" → "interesting" or "surprising"

### Absolute Statements
- "proves" → "demonstrates"
- "all methods fail" → "methods we tested fail"
- "cannot work" → "do not work in our experiments"

### Emotional Appeals
- Remove "sobering result"
- Avoid "troubling" when possible
- Let readers draw emotional conclusions

## 8. Example Transformations

Bad: "This shocking discovery exposes the OOD illusion"
Good: "This discovery led us to a deeper realization"

Bad: "Methods catastrophically explode"
Good: "Methods amplify errors by a considerable margin"

Bad: "This proves current methods are fundamentally flawed"
Good: "This suggests we need to radically rethink our approach"

## 9. Statistical Rigor (From Reviewer Feedback)

### Support Claims with Statistics
- Replace "no method improves" with "no tested method showed statistically significant improvement (p > 0.05)"
- Include confidence intervals: "235% worse (95% CI: 220-250%)"
- Report standard errors in all tables
- Add error bars to all figures

### Multiple Seeds and Reproducibility
- Always report number of seeds: "(n=5 seeds)"
- Include per-seed results in appendix
- Report mean ± std for all metrics
- Test statistical significance for key claims

## 10. Presentation Style (From Reviewer Feedback)

### Avoid Rhetorical Questions
- Bad: "Consider two scenarios: How would you adapt?"
- Good: "We examine two scenarios that illustrate different types of distribution shift"

### Don't Repeat Key Numbers
- Mention "235% degradation" prominently once
- Use variations in other mentions: "substantial degradation", "the observed performance drop"
- Let readers remember impactful numbers without hammering them

### Proper Figure/Table References
- Every figure/table must be referenced in text
- Use: "As shown in Table 2..." or "Figure 3 demonstrates..."
- Number all equations and reference them: "Using Equation (3)..."
- Include informative captions that can stand alone

## 11. Citation Practices (From Reviewer Feedback)

### Include Recent Work
- Prioritize 2023-2025 papers to show current relevance
- Position your work within contemporary conversation
- Acknowledge recent advances even if they don't solve your specific problem
- Example: "While PeTTA [2024] addresses collapse, mechanism shifts require..."

### Balanced Citations
- Cite work that supports AND challenges your thesis
- Explain why apparently contradictory work doesn't apply
- Give credit to partial solutions: "X addresses Y but not Z"

### Complete Bibliography
- Include DOIs or arXiv IDs for all citations
- Use consistent formatting (follow venue guidelines)
- Ensure bidirectional integrity (every citation appears in text and bibliography)
- Aim for 40-50 references for full papers

## 12. Speculation and Future Work (From Reviewer Feedback)

### Move Speculation to Dedicated Sections
- Bad: "We speculate this might be because..." (in Results)
- Good: Create explicit "Future Work" or "Open Questions" subsection
- Keep Results/Analysis sections evidence-based

### Label Uncertainty Clearly
- "One possible explanation..." → Move to Discussion
- "This might suggest..." → Support with evidence or move to Future Work
- "We hypothesize..." → Need experimental validation or clearly label as untested

## 13. Scope Management (From Reviewer Feedback)

### Calibrate Claims to Evidence
- If you tested one task, don't claim universal failure
- Scope to your domain: "In physics tasks with mechanism shifts..."
- Acknowledge boundary of your findings: "While our results apply to X, Y remains open"

### Use Precise Language for Scope
- "The studied setting" not "all settings"
- "Physics mechanism shifts" not "OOD in general"
- "Self-supervised adaptation" not "all adaptation methods"
- "Our experiments show" not "This proves"

## 14. Professional Polish

### Consistent Terminology
- Define terms once, use consistently
- Create glossary if many technical terms
- Avoid switching between synonyms (pick "mechanism shift" OR "generative process change")

### Mathematical Notation
- Define all symbols when first used
- Use consistent notation throughout
- Number all important equations
- Explain equations in words too

### Supplementary Materials
- Put extensive ablations in appendix
- Include implementation details
- Provide code/data availability statement
- Add reproducibility checklist

## Example Review Response
When addressing reviewer feedback:
- Thank reviewer for constructive comments
- Address each point systematically
- Explain what changed with specific line numbers
- Show how feedback improved the work
