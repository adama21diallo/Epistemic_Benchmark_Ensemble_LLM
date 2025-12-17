# Epistemic Bundles: Diagnosing Reliability in LLM Ensembles

### Abstract
Ensemble methods for Large Language Models (LLMs) are commonly framed as **selection problems**: given multiple candidate outputs, the system attempts to vote, rerank, or verify the best one. This relies on the _Selection Hypothesis_—the assumption that the complete correct answer is present among the candidates.

In reasoning-heavy tasks, this assumption frequently fails. Truth is often fragmented across outputs, partially correct, or entangled with hallucinated steps.

We introduce a framework for **Epistemic Integration**, which characterizes ensemble behavior across five epistemic states:
1.  **Redundant Correctness**
2.  **Complementary Fragmentation**
3.  **Reasoning–Result Divergence**
4.  **Contradictory Hallucination**
5.  **False Consensus**

To operationalize this, we propose **Epistemic Bundles**, a general methodology for constructing diagnostic benchmarks by algorithmically perturbing reference solutions. We present **Epistemic-GSM**, a dataset of 8,500 bundles derived from GSM8K.

We argue that true reliability requires systems that can **reconstruct truth from fragments** and, crucially, **recognize when evidence is insufficient** (abstention) rather than forcing a consensus.
