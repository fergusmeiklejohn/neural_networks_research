# Review Response Progress Summary

## ✅ Completed Work

### 1. Pendulum Experiment (Addresses "Broaden Empirical Base")
- ✅ Created pendulum data generator with time-varying length mechanism
- ✅ Tested data generation and physics correctness
- ✅ Adapted baseline models for pendulum state representation
- ✅ Ran quick baseline training - found 1.4x degradation (much milder than 2-ball)
- ✅ Implemented physics-aware TTA variants (energy, Hamiltonian consistency)
- ✅ **KEY FINDING**: Even physics-aware TTA degrades performance (12-18x)
- ✅ Created comparison summary showing TTA fails across different mechanism types

### 2. Style Guidelines Update
- ✅ Created comprehensive `SCIENTIFIC_WRITING_NOTES.md` incorporating reviewer feedback
- ✅ Updated `CLAUDE.md` to reference style guide for all paper writing
- ✅ Included specific examples of good vs bad phrasing

### 3. Planning Documents Created
- ✅ `review_response_plan.md` - Overall strategy
- ✅ `pendulum_experiment_plan.md` - Second mechanism-shift task design
- ✅ `literature_integration_plan.md` - How to incorporate recent papers
- ✅ `claim_tempering_revisions.md` - Specific edits to narrow claims
- ✅ `review_1_megathinking.md` - Comprehensive synthesis

## 📋 Remaining Tasks

### High Priority (Must Do)

1. **Update Paper Draft**
   - [ ] Add pendulum results to experiments section
   - [ ] Integrate PeTTA, TAIP, TTAB citations and discussion
   - [ ] Implement all claim tempering revisions
   - [ ] Add statistical significance (p-values, CIs) to all results
   - [ ] Create proper figures with error bars

2. **Literature Integration**
   - [ ] Add new Background subsection: "Recent Advances in Stabilized TTA"
   - [ ] Add Discussion subsection: "Relationship to Contemporary TTA"
   - [ ] Update Introduction to acknowledge recent progress
   - [ ] Ensure 40-50 references with DOIs

3. **Statistical Rigor**
   - [ ] Calculate 95% confidence intervals for all metrics
   - [ ] Add error bars to all plots
   - [ ] Report p-values for key comparisons
   - [ ] Use "no significant improvement (p > 0.05)" instead of absolutes

### Medium Priority (Should Do)

4. **Additional Experiments**
   - [ ] Run full pendulum training with 10k samples
   - [ ] Consider damped oscillator as third mechanism type
   - [ ] Test PeTTA's collapse detection on our tasks

5. **Figures and Visualization**
   - [ ] Figure 1: Mechanism shift vs statistical shift schematic
   - [ ] Figure 2: Time-varying examples (gravity, pendulum length)
   - [ ] Figure 3: Performance degradation with error bars
   - [ ] Figure 4: Gradient alignment visualization

### Low Priority (Nice to Have)

6. **Extended Analysis**
   - [ ] Investigate why pendulum shows milder degradation
   - [ ] Add representation space analysis for pendulum
   - [ ] Test discrete mechanism changes (piecewise gravity)

## Key Messages for Revision

### Core Narrative
"We identify mechanism shifts as a specific class of distribution shift where current methods—including recent advances—show systematic failure. This complements existing work by delineating the boundaries of current approaches."

### Positioning
- Not contradicting recent advances (PeTTA, TAIP)
- Identifying a specific, persistent challenge
- Proposing diagnostic tools (gradient alignment)
- Opening new research directions

### Scope
- Always qualify: "physics tasks with mechanism shifts"
- Acknowledge: "self-supervised adaptation methods"
- Clarify: "when generative equations change"

## Next Immediate Steps

1. **Start Paper Revisions**
   - Open `full_paper_draft.md`
   - Implement claim tempering throughout
   - Add pendulum results section

2. **Literature Integration**
   - Add citations for PeTTA, TAIP, TTAB
   - Write new background subsection
   - Update introduction framing

3. **Statistical Updates**
   - Calculate CIs for existing results
   - Update all tables with significance tests
   - Create error bar plots

## Response to Reviewer Template

"We thank the reviewer for their constructive feedback. We have:

1. **Broadened our empirical base** by adding pendulum experiments with time-varying length. Results confirm TTA degradation (12-18x) even with physics-aware losses.

2. **Tested physics-aware TTA variants** including energy and Hamiltonian consistency. These still degrade performance, confirming the issue is fundamental to mechanism shifts.

3. **Integrated recent literature** including PeTTA, TAIP, and TTAB, positioning our work as identifying a specific challenge that persists despite recent advances.

4. **Tempered our claims** to specifically address 'physics tasks with mechanism shifts' rather than universal statements about OOD.

5. **Added statistical rigor** with confidence intervals, p-values, and error bars throughout.

These revisions strengthen our contribution while properly contextualizing it within the broader landscape of OOD research."