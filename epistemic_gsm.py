import re
import random
import json
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Any

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
OUTPUT_FILE = "epistemic_gsm.jsonl"
NUM_CANDIDATES = 5  # Size of the ensemble (A) per bundle

random.seed(SEED)
np.random.seed(SEED)

class EpistemicPerturber:
    """
    Implements the perturbation logic described in the paper:
    Reasoning Erasure, Answer Swapping, Noise Injection, Fragmentation.
    """

    @staticmethod
    def extract_answer(solution: str) -> str:
        """Extracts the numeric answer after '####'."""
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution)
        return match.group(1).replace(',', '') if match else None

    @staticmethod
    def get_steps(solution: str) -> List[str]:
        """Splits reasoning into sentences/steps."""
        # Remove the final answer part
        reasoning = solution.split("####")[0]
        # Split by newlines or periods followed by spaces
        steps = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', reasoning) if s.strip()]
        return steps

    @staticmethod
    def generate_wrong_answer(correct_ans: str) -> str:
        """Perturbs the numeric answer (e.g., +10%, off-by-one)."""
        try:
            val = float(correct_ans)
            if val.is_integer():
                # Random integer shift
                offset = random.choice([-1, 1, random.randint(-10, 10)])
                return str(int(val + offset))
            else:
                # Float shift
                return f"{val * random.uniform(0.8, 1.2):.2f}"
        except:
            return "0"

    @staticmethod
    def create_divergence(solution: str, steps: List[str], correct_ans: str) -> str:
        """Class III: Keeps reasoning correct, swaps final answer."""
        wrong_ans = EpistemicPerturber.generate_wrong_answer(correct_ans)
        # Reconstruct valid reasoning but append wrong answer
        return " ".join(steps) + f"\n#### {wrong_ans}"

    @staticmethod
    def create_fragment(steps: List[str], start_ratio: float, end_ratio: float) -> str:
        """Class II: Returns a slice of the reasoning steps."""
        n = len(steps)
        start = int(n * start_ratio)
        end = int(n * end_ratio)
        fragment = steps[start:end]
        if not fragment: return "..." # Fallback for very short solutions
        return " ".join(fragment) + " [Truncated]"

    @staticmethod
    def create_hallucination(other_solutions: List[str]) -> str:
        """Class IV: Returns a completely irrelevant solution (Cross-Sample Pollution)."""
        # Pick a solution from a different problem to simulate confident hallucination
        hallucination = random.choice(other_solutions)
        # Modify the answer to be consistent with the hallucination but wrong for the query
        return hallucination

# ==========================================
# BUNDLE GENERATION LOGIC
# ==========================================

class BundleFactory:
    def __init__(self, dataset):
        self.dataset = dataset
        # Cache all solutions for hallucination injection
        self.all_solutions = [x['answer'] for x in dataset]

    def create_class_i(self, idx, row) -> Dict:
        """Class I: Redundant Correctness (At least one fully correct)."""
        candidates = []
        # 1. The Truth
        candidates.append(row['answer'])
        # 2. A duplicate truth
        candidates.append(row['answer'])
        # 3-5. Noise/Wrong
        for _ in range(NUM_CANDIDATES - 2):
            candidates.append(EpistemicPerturber.create_hallucination(self.all_solutions))
        
        random.shuffle(candidates)
        return self._pack(row, candidates, "Class I: Redundant Correctness")

    def create_class_ii(self, idx, row) -> Dict:
        """Class II: Complementary Fragmentation (Truth split across candidates)."""
        steps = EpistemicPerturber.get_steps(row['answer'])
        candidates = []
        
        # Create overlapping fragments that cover the whole chain
        # Candidate 1: 0% to 60%
        candidates.append(EpistemicPerturber.create_fragment(steps, 0.0, 0.6))
        # Candidate 2: 40% to 100% (but missing the final formatting)
        candidates.append(EpistemicPerturber.create_fragment(steps, 0.4, 1.0))
        # Candidate 3: Middle chunk
        candidates.append(EpistemicPerturber.create_fragment(steps, 0.3, 0.7))
        
        # Fill rest with noise
        for _ in range(NUM_CANDIDATES - 3):
            candidates.append(EpistemicPerturber.create_hallucination(self.all_solutions))
            
        random.shuffle(candidates)
        return self._pack(row, candidates, "Class II: Complementary Fragmentation")

    def create_class_iii(self, idx, row) -> Dict:
        """Class III: Reasoning-Result Divergence (Correct logic, wrong answer)."""
        steps = EpistemicPerturber.get_steps(row['answer'])
        ans = EpistemicPerturber.extract_answer(row['answer'])
        candidates = []
        
        # All candidates have perfect reasoning but different WRONG answers
        for _ in range(NUM_CANDIDATES):
            candidates.append(EpistemicPerturber.create_divergence(row['answer'], steps, ans))
            
        return self._pack(row, candidates, "Class III: Reasoning-Result Divergence")

    def create_class_iv(self, idx, row) -> Dict:
        """Class IV: Contradictory Hallucination (All wrong, high variance)."""
        candidates = []
        # Inject solutions from completely different math problems
        # This simulates high-confidence hallucinations that are logically sound but irrelevant
        for _ in range(NUM_CANDIDATES):
            candidates.append(EpistemicPerturber.create_hallucination(self.all_solutions))
            
        return self._pack(row, candidates, "Class IV: Contradictory Hallucination")

    def create_class_v(self, idx, row) -> Dict:
        """Class V: False Consensus (All agree on the SAME wrong answer)."""
        candidates = []
        ans = EpistemicPerturber.extract_answer(row['answer'])
        wrong_ans = EpistemicPerturber.generate_wrong_answer(ans)
        steps = EpistemicPerturber.get_steps(row['answer'])
        
        # Generate a "False Consensus" output
        consensus_output = " ".join(steps) + f"\n#### {wrong_ans}"
        
        for _ in range(NUM_CANDIDATES):
            candidates.append(consensus_output)
            
        return self._pack(row, candidates, "Class V: False Consensus")

    def _pack(self, row, candidates, label) -> Dict:
        return {
            "question": row['question'],
            "ground_truth": row['answer'],
            "candidates": candidates,
            "epistemic_class": label
        }

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main", split="test") # Using test set for generation
    factory = BundleFactory(ds)
    
    bundles = []
    print(f"Generating bundles for {len(ds)} examples...")
    
    for i, row in enumerate(ds):
        # Cyclically assign classes to ensure balanced dataset
        mode = i % 5
        if mode == 0:
            bundle = factory.create_class_i(i, row)
        elif mode == 1:
            bundle = factory.create_class_ii(i, row)
        elif mode == 2:
            bundle = factory.create_class_iii(i, row)
        elif mode == 3:
            bundle = factory.create_class_iv(i, row)
        elif mode == 4:
            bundle = factory.create_class_v(i, row)
            
        bundles.append(bundle)

    # Save to JSONL
    print(f"Saving {len(bundles)} bundles to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for b in bundles:
            f.write(json.dumps(b) + "\n")
            
    print("Done! Dataset ready for GitHub.")

if __name__ == "__main__":
    main()
