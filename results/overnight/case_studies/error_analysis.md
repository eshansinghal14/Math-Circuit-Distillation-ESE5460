# Error analysis

## 22_add

Accuracy by structure:

| stratum | n | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2000 | 0.960 | 0.670 | 0.993 | 0.981 | 1.000 |
| 0 carries | 601 | 0.965 | 0.529 | 0.998 | 0.977 | 0.998 |
| 1 carry | 929 | 0.953 | 0.684 | 0.994 | 0.985 | 1.000 |
| 2+ carries | 470 | 0.966 | 0.823 | 0.985 | 0.977 | 1.000 |
| new digit | 964 | 0.942 | 0.810 | 0.989 | 0.981 | 1.000 |
| no new digit | 1036 | 0.976 | 0.540 | 0.997 | 0.980 | 0.999 |
| 1-digit operand | 372 | 0.965 | 0.613 | 0.997 | 0.968 | 0.997 |

Wrong answers by type (count, share of that model's errors):

| error | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- |
| format | 75 (93%) | 571 (87%) | 0 (0%) | 15 (38%) | 0 (0%) |
| copy_operand | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 1 (100%) |
| carry_dropped | 0 (0%) | 9 (1%) | 5 (36%) | 3 (8%) | 0 (0%) |
| carry_extra | 0 (0%) | 16 (2%) | 1 (7%) | 7 (18%) | 0 (0%) |
| transposition | 0 (0%) | 0 (0%) | 0 (0%) | 1 (3%) | 0 (0%) |
| units_wrong | 6 (7%) | 61 (9%) | 7 (50%) | 13 (33%) | 0 (0%) |
| tens_wrong | 0 (0%) | 3 (0%) | 1 (7%) | 0 (0%) | 0 (0%) |
| high_wrong | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **wrong total** | 81 | 660 | 14 | 39 | 1 |

## 222_add

Accuracy by structure:

| stratum | n | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2000 | 0.882 | 0.199 | 0.425 | 0.431 | 0.477 |
| 0 carries | 112 | 0.973 | 0.277 | 0.670 | 0.661 | 0.786 |
| 1 carry | 611 | 0.894 | 0.221 | 0.491 | 0.491 | 0.550 |
| 2+ carries | 1277 | 0.868 | 0.182 | 0.373 | 0.382 | 0.416 |
| new digit | 1628 | 0.863 | 0.184 | 0.391 | 0.400 | 0.434 |
| no new digit | 372 | 0.965 | 0.263 | 0.578 | 0.565 | 0.667 |
| 1-digit operand | 556 | 0.878 | 0.290 | 0.568 | 0.606 | 0.685 |

Wrong answers by type (count, share of that model's errors):

| error | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- |
| format | 179 (76%) | 484 (30%) | 0 (0%) | 48 (4%) | 0 (0%) |
| copy_operand | 0 (0%) | 11 (1%) | 1 (0%) | 18 (2%) | 4 (0%) |
| carry_dropped | 10 (4%) | 53 (3%) | 298 (26%) | 276 (24%) | 323 (31%) |
| carry_extra | 0 (0%) | 192 (12%) | 118 (10%) | 218 (19%) | 226 (22%) |
| transposition | 0 (0%) | 4 (0%) | 8 (1%) | 3 (0%) | 1 (0%) |
| units_wrong | 8 (3%) | 565 (35%) | 588 (51%) | 385 (34%) | 289 (28%) |
| tens_wrong | 39 (17%) | 293 (18%) | 136 (12%) | 190 (17%) | 202 (19%) |
| high_wrong | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **wrong total** | 236 | 1602 | 1149 | 1138 | 1045 |

## 2222_add

Accuracy by structure:

| stratum | n | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2000 | 0.770 | 0.046 | 0.082 | 0.116 | 0.129 |
| 0 carries | 4 | 1.000 | 0.000 | 0.500 | 0.500 | 0.500 |
| 1 carry | 197 | 0.883 | 0.086 | 0.152 | 0.213 | 0.223 |
| 2+ carries | 1799 | 0.757 | 0.042 | 0.073 | 0.104 | 0.117 |
| new digit | 1918 | 0.764 | 0.044 | 0.074 | 0.106 | 0.122 |
| no new digit | 82 | 0.915 | 0.110 | 0.268 | 0.329 | 0.280 |
| 1-digit operand | 673 | 0.810 | 0.071 | 0.126 | 0.149 | 0.174 |

Wrong answers by type (count, share of that model's errors):

| error | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- |
| format | 56 (12%) | 519 (27%) | 0 (0%) | 10 (1%) | 0 (0%) |
| copy_operand | 0 (0%) | 11 (1%) | 0 (0%) | 63 (4%) | 52 (3%) |
| carry_dropped | 41 (9%) | 32 (2%) | 165 (9%) | 167 (9%) | 214 (12%) |
| carry_extra | 14 (3%) | 103 (5%) | 67 (4%) | 117 (7%) | 127 (7%) |
| transposition | 0 (0%) | 12 (1%) | 15 (1%) | 5 (0%) | 9 (1%) |
| units_wrong | 159 (35%) | 978 (51%) | 1361 (74%) | 1116 (63%) | 1003 (58%) |
| tens_wrong | 190 (41%) | 252 (13%) | 229 (12%) | 291 (16%) | 338 (19%) |
| high_wrong | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **wrong total** | 460 | 1907 | 1837 | 1769 | 1743 |

## 33_add

Accuracy by structure:

| stratum | n | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2000 | 0.922 | 0.487 | 0.658 | 0.579 | 0.659 |
| 0 carries | 359 | 0.889 | 0.652 | 0.925 | 0.819 | 0.925 |
| 1 carry | 723 | 0.921 | 0.505 | 0.714 | 0.638 | 0.719 |
| 2+ carries | 918 | 0.937 | 0.410 | 0.511 | 0.439 | 0.509 |
| new digit | 1004 | 0.945 | 0.303 | 0.415 | 0.342 | 0.397 |
| no new digit | 996 | 0.900 | 0.674 | 0.904 | 0.818 | 0.924 |
| 1-digit operand | 43 | 0.744 | 0.488 | 0.953 | 0.791 | 0.977 |

Wrong answers by type (count, share of that model's errors):

| error | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- |
| format | 112 (72%) | 107 (10%) | 0 (0%) | 10 (1%) | 0 (0%) |
| copy_operand | 3 (2%) | 7 (1%) | 0 (0%) | 9 (1%) | 2 (0%) |
| carry_dropped | 9 (6%) | 141 (14%) | 168 (25%) | 163 (19%) | 187 (27%) |
| carry_extra | 6 (4%) | 185 (18%) | 97 (14%) | 138 (16%) | 132 (19%) |
| transposition | 0 (0%) | 14 (1%) | 12 (2%) | 14 (2%) | 11 (2%) |
| units_wrong | 8 (5%) | 290 (28%) | 265 (39%) | 240 (29%) | 123 (18%) |
| tens_wrong | 4 (3%) | 250 (24%) | 109 (16%) | 231 (27%) | 185 (27%) |
| high_wrong | 13 (8%) | 31 (3%) | 32 (5%) | 37 (4%) | 41 (6%) |
| **wrong total** | 155 | 1025 | 683 | 842 | 681 |

## 21_mult

Accuracy by structure:

| stratum | n | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1000 | 0.894 | 0.124 | 0.669 | 0.404 | 0.949 |
| 0 carries | 270 | 0.919 | 0.381 | 0.907 | 0.859 | 0.963 |
| 1 carry | 266 | 0.955 | 0.038 | 0.579 | 0.297 | 0.940 |
| 2+ carries | 464 | 0.845 | 0.024 | 0.582 | 0.200 | 0.946 |
| new digit | 672 | 0.875 | 0.030 | 0.571 | 0.229 | 0.942 |
| no new digit | 328 | 0.933 | 0.317 | 0.869 | 0.762 | 0.963 |

Wrong answers by type (count, share of that model's errors):

| error | Teacher 8B | Base 1B | SFT | Standard KD | Graph KD |
| --- | --- | --- | --- | --- | --- |
| format | 93 (88%) | 872 (100%) | 116 (35%) | 579 (97%) | 1 (2%) |
| copy_operand | 3 (3%) | 0 (0%) | 39 (12%) | 1 (0%) | 2 (4%) |
| carry_dropped | 1 (1%) | 0 (0%) | 2 (1%) | 0 (0%) | 0 (0%) |
| carry_extra | 0 (0%) | 0 (0%) | 0 (0%) | 1 (0%) | 1 (2%) |
| transposition | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| units_wrong | 5 (5%) | 3 (0%) | 114 (34%) | 12 (2%) | 34 (67%) |
| tens_wrong | 4 (4%) | 1 (0%) | 54 (16%) | 3 (1%) | 13 (25%) |
| high_wrong | 0 (0%) | 0 (0%) | 6 (2%) | 0 (0%) | 0 (0%) |
| **wrong total** | 106 | 876 | 331 | 596 | 51 |
