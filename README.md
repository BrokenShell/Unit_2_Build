# Algorithm Prediction Research
### Random Distribution Detection Project
##### by Robert Sharp
<br/>

## Target Algorithms:
- front_linear
- back_linear
- front_gauss
- back_gauss
- front_poisson
- back_poisson

## Distribution Ranges:
- d4 `[1..4]`
- d6 `[1..6]`
- d8 `[1..8]`
- d10 `[1..10]`
- d12 `[1..12]`
- d20 `[1..20]`

## Data Sets:
Each set contains 10,000 rows of 10 random rolls of a random distribution algorithm over a given range. A Flat Uniform Distribution is used to select the algorithm for each row.
- dice_4.csv
- dice_6.csv
- dice_8.csv
- dice_10.csv
- dice_12.csv
- dice_20.csv

## Features:
A series of 10 random rolls of a given range, the specific distribution is produced with a random algorithm.

## Baseline Guess:
For a given a range, one would have a 1 in 6 chance (16.66%) to guess the correct algorithm.

## Model & Training
RandomForestClassifier. Six models will be trained to recognize 6 algorithms across 6 data sets.

## Research Question: 
_Are smaller dice more difficult to predict?_
- TL;DR: Mostly Yes.
