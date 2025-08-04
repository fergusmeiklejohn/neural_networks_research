# 8. Conclusion

We have shown that on physics tasks involving mechanism shifts—where the data-generating equations change—current self-supervised adaptation methods show systematic performance degradation. Test-time adaptation increased error by 235% on time-varying gravity and 12-18x on variable-length pendulum tasks, even when using physics-aware losses. Our implementation of collapse detection inspired by PeTTA successfully prevented degenerate solutions but provided negligible accuracy improvement (0.06%), demonstrating that stability and accuracy are orthogonal challenges for mechanism shifts.

These empirical results across two different physics systems reveal a fundamental limitation: when test distributions require new computational operations (such as the L̇/L term in variable pendulum dynamics), parameter adaptation within fixed architectures cannot bridge this gap. Our gradient alignment analysis explains why—self-supervised objectives become negatively aligned with true prediction error under mechanism shifts, causing adaptation to move away from accurate solutions.

## Positioning Within Current Research

Our findings complement rather than contradict recent advances in test-time adaptation. Methods like PeTTA excel at preventing adaptation collapse, while TAIP succeeds when physical laws remain fixed but parameters vary. TTAB's comprehensive analysis shows no single method handles all distribution shifts. We extend this understanding by identifying mechanism shifts as a specific failure mode where new computational operations are required—a challenge distinct from parameter adaptation or stability preservation.

This delineation helps clarify when different approaches are appropriate:
- **Parameter shifts** (different constants): TAIP and similar physics-aware methods excel
- **Stability challenges** (risk of collapse): PeTTA's monitoring prevents degradation
- **Mechanism shifts** (new physics terms): Current methods reach their limits

## Implications for Future Research

Our analysis suggests several directions for handling mechanism shifts:

1. **Modular architectures** that can activate dormant computational pathways or reconfigure connections to express new operations

2. **Program synthesis at test time** to discover and implement new functional forms from observed data

3. **Hybrid approaches** that detect when parameter adaptation fails and switch to structure learning

4. **Benchmarks explicitly designed for mechanism shifts** beyond physics, such as reasoning tasks where solution strategies change

## Practical Considerations

For practitioners facing distribution shifts, we recommend:
- Diagnose whether shifts involve parameter changes or mechanism changes
- For parameter shifts, current TTA methods including physics-aware variants may help
- For suspected mechanism shifts, baseline models may outperform adaptation
- Monitor for both collapse (PeTTA-style) and accuracy degradation (our gradient alignment)

## Limitations and Scope

We acknowledge several limitations:
- Our experiments focus on physics prediction tasks with specific types of mechanism shifts
- We tested representative methods but not all possible architectures or adaptation strategies
- The boundary between parameter and mechanism shifts may vary across domains
- We implemented collapse detection inspired by PeTTA but not their exact algorithm

## Final Thoughts

This work identifies mechanism shifts in physics as a concrete challenge where current self-supervised adaptation methods fail despite recent advances. By demonstrating this across multiple systems with comprehensive testing—including physics-aware losses and collapse detection—we hope to inspire development of methods that can handle changes in computational requirements, not just parameter values.

The ability to adapt to new mechanisms has implications beyond physics prediction, from climate models encountering tipping points to financial systems experiencing regime changes. Understanding the boundaries of current methods is essential for their safe deployment and for directing research toward the remaining challenges in out-of-distribution generalization.
