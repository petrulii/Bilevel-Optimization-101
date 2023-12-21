# Bilevel Optimization
## Implicit Differentiation

A project on bilevel optimization in the context of work at Thoth team INRIA under the supervision of Michael Arbel.

Make sure to add BilevelProblem and InnerSolution directories to system PATH before running.

To install dependencies:
pip install -r dependencies.txt

To run the dsprites experiment:
python tests/Dsprites.py

To remove wandb logging:
uncomment lines 18-19 in file bilevel-optimization/src/tests/Dsprites.py