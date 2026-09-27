# Frozen POST condition-value range audit

Legal (0,1] input mass does not imply membership in the original TRAIN support. Empirical TRAIN min/max are descriptive, not rejection rules or guarantees of reliability. Directions were not selected by these statistics. Extreme range alone cannot establish the cause of reverse output responses.

Records use the original checkpoint FP32 buffers, FP32 log/clamp/standardize and frozen scalar gain, matching standardized_mass exactly. The interface-crossing coordinates themselves remain FP64.

No cross-cell output was read for this audit. No observations are filtered and no thresholds are fitted. Log-odds is a bounded coordinate, not a calibrated identity probability.

| Fold | TRAIN queries/pairs | Actual condition min/max | Frozen gain |
|---|---|---|---|
| 0 | 457/58496 | -220.82069 / 314.12674 | 59.86651819 |
| 1 | 454/58112 | -220.77573 / 324.27115 | 59.86651819 |
| 2 | 451/57728 | -210.62822 / 316.52319 | 59.86651819 |
| 3 | 461/59008 | -218.81985 / 318.90930 | 59.86651819 |
| 4 | 457/58496 | -218.08315 / 320.42911 | 59.86651819 |

| Scope | Cell | Candidate | Actual condition min/max | Beyond fold TRAIN min/max (values) | Beyond same-candidate LL/GG range (values) |
|---|---|---|---|---|---|
| jp128/log_M/J_with_P | GG | fixed_wrong | -232.77589 / 285.52243 | 1/120 | 0/120 |
| jp128/log_M/J_with_P | GG | target | -125.28651 / 293.43549 | 0/120 | 0/120 |
| jp128/log_M/J_with_P | GL | fixed_wrong | -113.19304 / 285.69299 | 0/120 | 12/120 |
| jp128/log_M/J_with_P | GL | target | -140.31512 / 288.20435 | 0/120 | 108/120 |
| jp128/log_M/J_with_P | LG | fixed_wrong | 151.19362 / 359.13101 | 20/99 | 12/99 |
| jp128/log_M/J_with_P | LG | target | 264.30017 / 428.12589 | 86/99 | 87/99 |
| jp128/log_M/J_with_P | LL | fixed_wrong | 255.08125 / 358.96515 | 78/120 | 0/120 |
| jp128/log_M/J_with_P | LL | target | 257.91318 / 368.39676 | 59/120 | 0/120 |
| jp128/log_M/J_without_P | GG | fixed_wrong | -287.69720 / 71.16695 | 10/120 | 0/120 |
| jp128/log_M/J_without_P | GG | target | -233.52844 / 102.14474 | 1/120 | 0/120 |
| jp128/log_M/J_without_P | GL | fixed_wrong | -223.28001 / 86.54109 | 1/120 | 21/120 |
| jp128/log_M/J_without_P | GL | target | -215.06433 / 86.77060 | 0/120 | 99/120 |
| jp128/log_M/J_without_P | LG | fixed_wrong | 141.17546 / 321.87177 | 2/120 | 21/120 |
| jp128/log_M/J_without_P | LG | target | 220.76234 / 388.78644 | 32/120 | 99/120 |
| jp128/log_M/J_without_P | LL | fixed_wrong | 231.44762 / 304.96848 | 0/120 | 0/120 |
| jp128/log_M/J_without_P | LL | target | 231.32553 / 310.24185 | 0/120 | 0/120 |
| jp128/log_M/P_with_J | GG | fixed_wrong | -232.77589 / 285.52243 | 1/120 | 0/120 |
| jp128/log_M/P_with_J | GG | target | -125.28651 / 293.43549 | 0/120 | 0/120 |
| jp128/log_M/P_with_J | GL | fixed_wrong | -137.28592 / 288.27652 | 0/120 | 87/120 |
| jp128/log_M/P_with_J | GL | target | -113.86312 / 301.86191 | 0/120 | 33/120 |
| jp128/log_M/P_with_J | LG | fixed_wrong | -436.01782 / 79.59343 | 38/120 | 87/120 |
| jp128/log_M/P_with_J | LG | target | -210.15166 / 93.71829 | 0/120 | 33/120 |
| jp128/log_M/P_with_J | LL | fixed_wrong | -287.69720 / 71.16695 | 10/120 | 0/120 |
| jp128/log_M/P_with_J | LL | target | -233.52844 / 102.14474 | 1/120 | 0/120 |
| jp128/log_M/P_without_J | GG | fixed_wrong | 255.08125 / 358.96515 | 78/120 | 0/120 |
| jp128/log_M/P_without_J | GG | target | 257.91318 / 368.39676 | 59/120 | 0/120 |
| jp128/log_M/P_without_J | GL | fixed_wrong | 256.57098 / 354.26669 | 65/120 | 47/120 |
| jp128/log_M/P_without_J | GL | target | 257.74805 / 362.76779 | 67/120 | 73/120 |
| jp128/log_M/P_without_J | LG | fixed_wrong | 224.78731 / 316.84360 | 1/120 | 47/120 |
| jp128/log_M/P_without_J | LG | target | 212.02487 / 315.87085 | 1/120 | 73/120 |
| jp128/log_M/P_without_J | LL | fixed_wrong | 231.44762 / 304.96848 | 0/120 | 0/120 |
| jp128/log_M/P_without_J | LL | target | 231.32553 / 310.24185 | 0/120 | 0/120 |
| jp128/log_odds_M/J_with_P | GG | fixed_wrong | -232.77589 / 285.52243 | 1/120 | 0/120 |
| jp128/log_odds_M/J_with_P | GG | target | -125.28651 / 293.43549 | 0/120 | 0/120 |
| jp128/log_odds_M/J_with_P | GL | fixed_wrong | -113.33244 / 285.66693 | 0/120 | 14/120 |
| jp128/log_odds_M/J_with_P | GL | target | -149.11693 / 288.62857 | 0/120 | 106/120 |
| jp128/log_odds_M/J_with_P | LG | fixed_wrong | 109.87068 / 349.95468 | 26/120 | 14/120 |
| jp128/log_odds_M/J_with_P | LG | target | 264.69675 / 411.37640 | 107/120 | 106/120 |
| jp128/log_odds_M/J_with_P | LL | fixed_wrong | 255.08125 / 358.96515 | 78/120 | 0/120 |
| jp128/log_odds_M/J_with_P | LL | target | 257.91318 / 368.39676 | 59/120 | 0/120 |
| jp128/log_odds_M/J_without_P | GG | fixed_wrong | -287.69720 / 71.16695 | 10/120 | 0/120 |
| jp128/log_odds_M/J_without_P | GG | target | -233.52844 / 102.14474 | 1/120 | 0/120 |
| jp128/log_odds_M/J_without_P | GL | fixed_wrong | -224.11061 / 86.54841 | 1/120 | 21/120 |
| jp128/log_odds_M/J_without_P | GL | target | -214.15160 / 86.84299 | 0/120 | 99/120 |
| jp128/log_odds_M/J_without_P | LG | fixed_wrong | 150.07027 / 317.39835 | 2/120 | 21/120 |
| jp128/log_odds_M/J_without_P | LG | target | 223.43631 / 355.20407 | 17/120 | 99/120 |
| jp128/log_odds_M/J_without_P | LL | fixed_wrong | 231.44762 / 304.96848 | 0/120 | 0/120 |
| jp128/log_odds_M/J_without_P | LL | target | 231.32553 / 310.24185 | 0/120 | 0/120 |
| jp128/log_odds_M/P_with_J | GG | fixed_wrong | -232.77589 / 285.52243 | 1/120 | 0/120 |
| jp128/log_odds_M/P_with_J | GG | target | -125.28651 / 293.43549 | 0/120 | 0/120 |
| jp128/log_odds_M/P_with_J | GL | fixed_wrong | -137.26311 / 286.93512 | 0/120 | 91/120 |
| jp128/log_odds_M/P_with_J | GL | target | -113.89168 / 298.58261 | 0/120 | 29/120 |
| jp128/log_odds_M/P_with_J | LG | fixed_wrong | -441.38086 / 77.83229 | 42/120 | 91/120 |
| jp128/log_odds_M/P_with_J | LG | target | -211.04942 / 97.39963 | 0/120 | 29/120 |
| jp128/log_odds_M/P_with_J | LL | fixed_wrong | -287.69720 / 71.16695 | 10/120 | 0/120 |
| jp128/log_odds_M/P_with_J | LL | target | -233.52844 / 102.14474 | 1/120 | 0/120 |
| jp128/log_odds_M/P_without_J | GG | fixed_wrong | 255.08125 / 358.96515 | 78/120 | 0/120 |
| jp128/log_odds_M/P_without_J | GG | target | 257.91318 / 368.39676 | 59/120 | 0/120 |
| jp128/log_odds_M/P_without_J | GL | fixed_wrong | 256.58524 / 355.42776 | 68/120 | 46/120 |
| jp128/log_odds_M/P_without_J | GL | target | 257.74326 / 362.01773 | 68/120 | 74/120 |
| jp128/log_odds_M/P_without_J | LG | fixed_wrong | 223.25388 / 318.82806 | 1/120 | 46/120 |
| jp128/log_odds_M/P_without_J | LG | target | 207.43752 / 318.97775 | 1/120 | 74/120 |
| jp128/log_odds_M/P_without_J | LL | fixed_wrong | 231.44762 / 304.96848 | 0/120 | 0/120 |
| jp128/log_odds_M/P_without_J | LL | target | 231.32553 / 310.24185 | 0/120 | 0/120 |
| phase9/log_M/native_amplitude | GG | fixed_wrong | -234.71165 / 223.56288 | 4/36 | 0/36 |
| phase9/log_M/native_amplitude | GG | target | -89.05379 / 257.38025 | 0/36 | 0/36 |
| phase9/log_M/native_amplitude | GL | fixed_wrong | -231.68980 / 223.66013 | 4/36 | 8/36 |
| phase9/log_M/native_amplitude | GL | target | -91.41409 / 258.42596 | 0/36 | 28/36 |
| phase9/log_M/native_amplitude | LG | fixed_wrong | -229.02007 / 225.25107 | 4/36 | 8/36 |
| phase9/log_M/native_amplitude | LG | target | -84.50389 / 259.72653 | 0/36 | 28/36 |
| phase9/log_M/native_amplitude | LL | fixed_wrong | -226.48387 / 226.39919 | 4/36 | 0/36 |
| phase9/log_M/native_amplitude | LL | target | -83.62788 / 258.49347 | 0/36 | 0/36 |
| phase9/log_M/permuted_amplitude | GG | fixed_wrong | -238.73277 / 205.89970 | 4/36 | 0/36 |
| phase9/log_M/permuted_amplitude | GG | target | -95.09897 / 244.93344 | 0/36 | 0/36 |
| phase9/log_M/permuted_amplitude | GL | fixed_wrong | -232.28090 / 206.66881 | 4/36 | 17/36 |
| phase9/log_M/permuted_amplitude | GL | target | -97.17219 / 244.49208 | 0/36 | 19/36 |
| phase9/log_M/permuted_amplitude | LG | fixed_wrong | -238.64160 / 202.26639 | 4/36 | 17/36 |
| phase9/log_M/permuted_amplitude | LG | target | -91.96235 / 242.52061 | 0/36 | 19/36 |
| phase9/log_M/permuted_amplitude | LL | fixed_wrong | -234.01717 / 204.16583 | 4/36 | 0/36 |
| phase9/log_M/permuted_amplitude | LL | target | -90.98872 / 240.37152 | 0/36 | 0/36 |
| phase9/log_odds_M/native_amplitude | GG | fixed_wrong | -234.71165 / 223.56288 | 4/36 | 0/36 |
| phase9/log_odds_M/native_amplitude | GG | target | -89.05379 / 257.38025 | 0/36 | 0/36 |
| phase9/log_odds_M/native_amplitude | GL | fixed_wrong | -231.84221 / 223.62659 | 4/36 | 9/36 |
| phase9/log_odds_M/native_amplitude | GL | target | -91.40776 / 258.39328 | 0/36 | 27/36 |
| phase9/log_odds_M/native_amplitude | LG | fixed_wrong | -228.86736 / 225.30754 | 4/36 | 9/36 |
| phase9/log_odds_M/native_amplitude | LG | target | -84.50398 / 259.60825 | 0/36 | 27/36 |
| phase9/log_odds_M/native_amplitude | LL | fixed_wrong | -226.48387 / 226.39919 | 4/36 | 0/36 |
| phase9/log_odds_M/native_amplitude | LL | target | -83.62788 / 258.49347 | 0/36 | 0/36 |
| phase9/log_odds_M/permuted_amplitude | GG | fixed_wrong | -238.73277 / 205.89970 | 4/36 | 0/36 |
| phase9/log_odds_M/permuted_amplitude | GG | target | -95.09897 / 244.93344 | 0/36 | 0/36 |
| phase9/log_odds_M/permuted_amplitude | GL | fixed_wrong | -232.03923 / 206.78171 | 4/36 | 19/36 |
| phase9/log_odds_M/permuted_amplitude | GL | target | -97.16832 / 244.39842 | 0/36 | 17/36 |
| phase9/log_odds_M/permuted_amplitude | LG | fixed_wrong | -238.88333 / 202.14758 | 4/36 | 19/36 |
| phase9/log_odds_M/permuted_amplitude | LG | target | -91.96143 / 242.52414 | 0/36 | 17/36 |
| phase9/log_odds_M/permuted_amplitude | LL | fixed_wrong | -234.01717 / 204.16583 | 4/36 | 0/36 |
| phase9/log_odds_M/permuted_amplitude | LL | target | -90.98872 / 240.37152 | 0/36 | 0/36 |

21 primary log-M J_with_P cross cells are explicitly invalid and were not transformed into actual adapter conditions. Their original M values remain in the JSON.
