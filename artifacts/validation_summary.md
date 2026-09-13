# Validation Summary

## Automated tests

`python -m unittest discover -s tests -v` -> **12/12 passed**.

## Default training

`python train.py --epochs 1` -> completed successfully on the compact default CPU configuration.

## Default experiment

`python run_experiment.py --epochs 1 --output artifacts/results.json` -> completed successfully and wrote the evaluation artifact.

## Music/playlist checks

The tests cover:

- deterministic lyrics tokenization;
- timestamp-safe playlist construction;
- playlist-weighted user music representations;
- music similarity between playlist-informed user state and candidate reel tracks;
- five-way dynamic gating including music;
- item cold-start through content-derived representations.

The supplied music data are synthetic and do not contain licensed song lyrics.
