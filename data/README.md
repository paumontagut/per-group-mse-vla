# Robot trial scores

`robot_trials.csv` contains the 60 real-robot scores reported in the
paper. Sixty trials, 20 per model, scored on a 4-point ordinal rubric.

| Column | Type | Meaning |
|--------|------|---------|
| `model`  | string | One of `pi05_baseline_80k`, `pi05_ours`, `hsr_smolvla_40k`. The `pi05_ours` row is our π₀.₅ fine-tune of the workshop baseline. |
| `trial`  | int | Trial index within the model, 1 to 20. |
| `score`  | int (1-4) | Rubric score. See below. |

## The 4-point rubric

The score is ordinal and was assigned by the operator during each
trial.

- **1.** The robot did not move towards the target object.
- **2.** The robot moved towards the object but did not attempt a grasp.
- **3.** The robot reached and touched the object but did not lift it.
- **4.** The robot completed the pick-up.

The rubric is ordinal, not interval. The distance between a 1 and a 2
is not the same as the distance between a 3 and a 4, which is why the
paper uses Mann-Whitney U (a non-parametric test that only relies on
ranks) and not a t-test.

## How the trials were run

Each model was evaluated against two objects (a ceramic mug and a
Cheez-It box) with 10 trials per object. The robot started from the
same pose every time, ~1 m in front of a white table, with the head
tilted down to keep the table in the centre of the head camera frame.
Object positions were drawn from a small predefined grid. The
instruction for each trial was a short natural-language command (for
example "pick up the mug") passed to the policy client as a launch
parameter, the same way the model received it during training. The
operator then watched the trial silently until either the task was
completed or the timer ran out.

The CSV does not separate by object because the per-object breakdown
was not part of the reported statistics. Counts per score level match
the paper exactly.

## Reproducing the statistics

```bash
python figures/robot_trials.py --csv data/robot_trials.csv
```

That prints the mean scores per model and the three pairwise
Mann-Whitney U p-values, and saves a figure to `figures/robot_trials.pdf`.
The numbers should match the paper to the third decimal.
