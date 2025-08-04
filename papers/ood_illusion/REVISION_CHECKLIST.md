# Paper Revision Checklist

## ✅ Completed

### Experimental Work
- [x] Pendulum data generator implemented
- [x] Pendulum baseline experiments run
- [x] Physics-aware TTA variants tested (energy, Hamiltonian)
- [x] Comparison between pendulum and 2-ball results

### Writing/Planning
- [x] Abstract revised with tempered claims
- [x] Literature review of PeTTA, TAIP, TTAB
- [x] Honest positioning strategy developed
- [x] Reviewer response drafted
- [x] Style guidelines updated in Claude.md

### Key Findings Documented
- [x] Physics-aware TTA fails on mechanism shifts (12-18x degradation)
- [x] Different mechanism types show different severity
- [x] Gradient alignment explains failure

## ⏳ Still Needed

### High Priority - Paper Integration

1. [ ] **Introduction Section**
   - [ ] Revise opening to focus on physics mechanism shifts
   - [ ] Add mentions of PeTTA, TAIP, TTAB as recent advances
   - [ ] Use tempered language throughout

2. [ ] **Background Section**
   - [ ] Insert new subsection on recent TTA advances
   - [ ] Position our work relative to recent papers

3. [ ] **Methods Section**
   - [ ] Add pendulum experimental setup details
   - [ ] Define "mechanism shift" precisely
   - [ ] Describe physics-aware TTA implementations

4. [ ] **Results Section**
   - [ ] Insert pendulum results with statistics
   - [ ] Add confidence intervals to all tables
   - [ ] Include p-values for comparisons

5. [ ] **Discussion Section**
   - [ ] Insert "Relationship to Contemporary TTA" subsection
   - [ ] Explain why physics-aware losses fail
   - [ ] Acknowledge what we didn't test

6. [ ] **Conclusion**
   - [ ] Scope claims appropriately
   - [ ] Emphasize complementary nature of findings

### Medium Priority - Polish

7. [ ] **Figures**
   - [ ] Create Figure 1: Mechanism vs statistical shift diagram
   - [ ] Add error bars to all plots
   - [ ] Create gradient alignment visualization

8. [ ] **References**
   - [ ] Complete bibliography (40-50 papers)
   - [ ] Add DOIs and proper formatting
   - [ ] Ensure all citations are bidirectional

9. [ ] **Statistical Analysis**
   - [ ] Calculate 95% CIs for all metrics
   - [ ] Run significance tests between methods
   - [ ] Document number of seeds/runs

### Final Steps

10. [ ] **Compile Full Paper**
    - [ ] Merge all sections into full_paper_draft.md
    - [ ] Check consistency of terminology
    - [ ] Verify all cross-references

11. [ ] **Final Review**
    - [ ] Check all claims are properly scoped
    - [ ] Ensure recent papers are fairly represented
    - [ ] Verify we acknowledge limitations

12. [ ] **Prepare Submission**
    - [ ] Format for venue requirements
    - [ ] Prepare supplementary materials
    - [ ] Write cover letter with revision summary

## Time Estimate

- Paper integration: 4-6 hours
- Statistical analysis: 2-3 hours
- Figures: 2-3 hours
- Final review: 2 hours
- **Total: 10-14 hours of focused work**

## Critical Path

1. Start with Introduction/Background updates (sets tone)
2. Add results with proper statistics
3. Update discussion/conclusion
4. Create figures in parallel
5. Final consistency pass

## Risk Mitigation

If time is short, prioritize:
1. Honest positioning in Introduction/Discussion
2. Pendulum results in main text
3. Basic statistics (means + CIs minimum)
4. Defer complex figures to appendix

The key is showing we:
- Tested physics-aware approaches
- Found consistent failures across systems
- Position honestly relative to recent work
- Acknowledge limitations explicitly
