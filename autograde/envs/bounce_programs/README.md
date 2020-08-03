Most of the JSON files are for testing purposes (validating the implementation of the environment).

In the `./broken_small`, we collect 10 broken environments, 5 used as training, 5 used as testing. 
Programs are disjoint -- meaning the testing environments are completely foreign to the video classifier.
But they share the same theme.

Realistic setting: 
Teacher's reference/correct implementation; Provide 5 possible broken implementations.

Evaluate on student programs -- different broken things or same.

RolloutGenerator: produce training video data.

RolloutEvaluator: takes programs, generate videos, and run inference on videos, then build a concensus algorithm for decision.