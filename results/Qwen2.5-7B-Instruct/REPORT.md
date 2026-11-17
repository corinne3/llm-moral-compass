# Results - Qwen/Qwen2.5-7B-Instruct

## 1. Measurement quality

- Position bias (P('1') - 0.5): **-0.037**
- Order consistency (same decision when cases are swapped): **50%**
- Format validity (mean probability on '1'/'2'): **0.87**
- Prompts: 3600
- What a single-order study would conclude about 'inaction': **+0.04** (stay shown as Case 1) or **+0.18** (stay shown as Case 2). Counterbalanced value: see the AMCE table below.

## 2. Baseline moral preferences (AMCE, 95% CI)

| preference                  |   amce |   ci_low |   ci_high |   n |
|:----------------------------|-------:|---------:|----------:|----:|
| Humans over pets            | -0.197 |   -0.251 |    -0.142 |  50 |
| More lives over fewer       |  0.128 |    0.083 |     0.178 |  50 |
| Young over elderly          | -0.05  |   -0.099 |    -0.003 |  50 |
| Lawful over jaywalking      | -0.047 |   -0.088 |    -0.008 |  94 |
| Higher over lower status    |  0.071 |    0.01  |     0.136 |  50 |
| Fit over large              |  0.053 |    0.003 |     0.105 |  50 |
| Females over males          |  0.138 |    0.093 |     0.179 |  50 |
| Pedestrians over passengers |  0.019 |   -0.027 |     0.062 |  78 |
| Inaction over swerving      |  0.11  |    0.088 |     0.132 | 300 |

![amce.png](amce.png)

### Which preferences survive a change of wording / instrument?

| preference                  |    min |   max | verdict            |
|:----------------------------|-------:|------:|:-------------------|
| Humans over pets            | -0.76  | 0.208 | depends on wording |
| More lives over fewer       | -0.972 | 0.635 | depends on wording |
| Young over elderly          | -0.425 | 0.618 | depends on wording |
| Lawful over jaywalking      | -0.654 | 0.365 | depends on wording |
| Higher over lower status    | -0.442 | 0.296 | depends on wording |
| Fit over large              | -0.117 | 0.477 | depends on wording |
| Females over males          |  0.039 | 0.36  | depends on wording |
| Pedestrians over passengers | -0.123 | 0.103 | depends on wording |
| Inaction over swerving      |  0.045 | 0.166 | robust +           |

### AMCE per wording (5 = answer by name)

| dimension     | 0                   | 1                   | 2                   | 3                   | 4                   | 5                   |
|:--------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| age           | -0.43 [-0.48,-0.37] | -0.39 [-0.45,-0.32] | -0.15 [-0.19,-0.10] | +0.12 [-0.01,+0.25] | -0.08 [-0.20,+0.03] | +0.62 [+0.51,+0.71] |
| fitness       | -0.11 [-0.18,-0.02] | -0.12 [-0.18,-0.06] | -0.07 [-0.12,-0.02] | +0.14 [+0.03,+0.24] | -0.01 [-0.12,+0.12] | +0.48 [+0.34,+0.61] |
| gender        | +0.16 [+0.08,+0.24] | +0.13 [+0.07,+0.18] | +0.04 [-0.01,+0.09] | +0.10 [+0.00,+0.19] | +0.04 [-0.09,+0.18] | +0.36 [+0.21,+0.50] |
| intervention  | +0.10 [+0.05,+0.15] | +0.04 [+0.00,+0.08] | +0.08 [+0.04,+0.11] | +0.14 [+0.09,+0.18] | +0.17 [+0.11,+0.22] | +0.14 [+0.06,+0.23] |
| law           | -0.11 [-0.19,-0.02] | -0.14 [-0.21,-0.07] | +0.06 [+0.00,+0.12] | +0.19 [+0.13,+0.25] | -0.65 [-0.72,-0.59] | +0.37 [+0.22,+0.51] |
| number        | +0.64 [+0.55,+0.71] | +0.42 [+0.34,+0.50] | +0.50 [+0.42,+0.57] | -0.18 [-0.27,-0.11] | +0.37 [+0.24,+0.51] | -0.97 [-1.00,-0.93] |
| relation      | -0.12 [-0.23,-0.01] | +0.08 [+0.01,+0.16] | +0.10 [+0.05,+0.16] | +0.01 [-0.08,+0.11] | +0.03 [-0.06,+0.11] | +0.01 [-0.15,+0.18] |
| social_status | +0.29 [+0.18,+0.41] | +0.17 [+0.09,+0.25] | +0.12 [+0.04,+0.20] | -0.01 [-0.10,+0.07] | +0.30 [+0.16,+0.43] | -0.44 [-0.60,-0.27] |
| species       | -0.11 [-0.21,-0.02] | -0.14 [-0.22,-0.06] | -0.08 [-0.14,-0.01] | -0.30 [-0.39,-0.21] | +0.21 [+0.08,+0.35] | -0.76 [-0.85,-0.66] |

### Comprehension check (does the model know who dies?)

|   template |   accuracy |   mean_p_correct |   mean_mass |   n |
|-----------:|-----------:|-----------------:|------------:|----:|
|          0 |      0.842 |            0.845 |           1 | 240 |
|          1 |      0.879 |            0.873 |           1 | 240 |
|          2 |      0.842 |            0.841 |           1 | 240 |
|          3 |      0.267 |            0.28  |           1 | 240 |
|          4 |      0.912 |            0.912 |           1 | 240 |

### Sensitivity to the wording

| dimension     |   template_0 |   template_1 |   template_2 |   template_3 |   template_4 |   template_5 |   spread_sd |
|:--------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|------------:|
| species       |       -0.111 |       -0.138 |       -0.077 |       -0.304 |        0.208 |       -0.76  |       0.322 |
| number        |        0.635 |        0.419 |        0.497 |       -0.182 |        0.374 |       -0.972 |       0.607 |
| age           |       -0.425 |       -0.391 |       -0.148 |        0.124 |       -0.081 |        0.618 |       0.386 |
| law           |       -0.107 |       -0.14  |        0.065 |        0.19  |       -0.654 |        0.365 |       0.352 |
| social_status |        0.292 |        0.169 |        0.124 |       -0.013 |        0.296 |       -0.442 |       0.277 |
| fitness       |       -0.107 |       -0.117 |       -0.066 |        0.138 |       -0.006 |        0.477 |       0.228 |
| gender        |        0.162 |        0.127 |        0.043 |        0.095 |        0.039 |        0.36  |       0.119 |
| relation      |       -0.123 |        0.078 |        0.103 |        0.015 |        0.026 |        0.015 |       0.079 |
| intervention  |        0.098 |        0.045 |        0.078 |        0.136 |        0.166 |        0.141 |       0.045 |

## 3. Prompting (significant changes only)

| condition             | dimension     |   delta |   ci_low |   ci_high |
|:----------------------|:--------------|--------:|---------:|----------:|
| talk_fear             | number        |  -0.437 |   -0.545 |    -0.326 |
| talk_fear             | age           |   0.284 |    0.181 |     0.382 |
| talk_fear             | law           |   0.285 |    0.209 |     0.364 |
| talk_fear             | social_status |  -0.159 |   -0.256 |    -0.06  |
| talk_fear             | intervention  |  -0.098 |   -0.16  |    -0.044 |
| talk_anger            | number        |  -0.413 |   -0.507 |    -0.321 |
| talk_anger            | age           |   0.271 |    0.187 |     0.344 |
| talk_anger            | law           |   0.233 |    0.164 |     0.304 |
| talk_anger            | social_status |  -0.13  |   -0.208 |    -0.042 |
| talk_anger            | intervention  |  -0.074 |   -0.132 |    -0.023 |
| talk_hope             | number        |  -0.42  |   -0.504 |    -0.336 |
| talk_hope             | age           |   0.274 |    0.192 |     0.345 |
| talk_hope             | law           |   0.218 |    0.147 |     0.291 |
| talk_hope             | social_status |  -0.106 |   -0.18  |    -0.022 |
| talk_hope             | intervention  |  -0.068 |   -0.122 |    -0.018 |
| persona_utilitarian   | law           |   0.417 |    0.337 |     0.499 |
| persona_utilitarian   | social_status |   0.169 |    0.017 |     0.341 |
| persona_utilitarian   | relation      |  -0.159 |   -0.224 |    -0.097 |
| persona_utilitarian   | intervention  |  -0.115 |   -0.175 |    -0.056 |
| persona_deontologist  | number        |  -0.314 |   -0.4   |    -0.228 |
| persona_deontologist  | age           |   0.158 |    0.082 |     0.228 |
| persona_deontologist  | law           |   0.121 |    0.059 |     0.186 |
| persona_deontologist  | social_status |  -0.158 |   -0.243 |    -0.08  |
| persona_deontologist  | intervention  |  -0.093 |   -0.139 |    -0.052 |
| persona_care_ethics   | species       |   0.199 |    0.107 |     0.281 |
| persona_care_ethics   | number        |  -0.216 |   -0.304 |    -0.125 |
| persona_care_ethics   | law           |   0.149 |    0.084 |     0.211 |
| persona_care_ethics   | intervention  |  -0.075 |   -0.118 |    -0.033 |
| country_France        | number        |  -0.362 |   -0.442 |    -0.284 |
| country_France        | age           |   0.207 |    0.124 |     0.281 |
| country_France        | law           |   0.21  |    0.146 |     0.274 |
| country_France        | social_status |  -0.104 |   -0.187 |    -0.024 |
| country_France        | intervention  |  -0.069 |   -0.121 |    -0.025 |
| country_Japan         | number        |  -0.334 |   -0.417 |    -0.249 |
| country_Japan         | age           |   0.165 |    0.083 |     0.236 |
| country_Japan         | law           |   0.194 |    0.133 |     0.253 |
| country_Japan         | social_status |  -0.094 |   -0.172 |    -0.014 |
| country_Japan         | intervention  |  -0.054 |   -0.103 |    -0.013 |
| country_United States | number        |  -0.284 |   -0.359 |    -0.206 |
| country_United States | age           |   0.149 |    0.066 |     0.22  |
| country_United States | law           |   0.168 |    0.109 |     0.224 |
| country_United States | intervention  |  -0.061 |   -0.107 |    -0.021 |
| country_Brazil        | number        |  -0.373 |   -0.462 |    -0.289 |
| country_Brazil        | age           |   0.232 |    0.139 |     0.313 |
| country_Brazil        | law           |   0.241 |    0.176 |     0.308 |
| country_Brazil        | social_status |  -0.123 |   -0.211 |    -0.038 |
| country_Brazil        | intervention  |  -0.077 |   -0.132 |    -0.03  |
| talk_neutral          | number        |  -0.381 |   -0.474 |    -0.286 |
| talk_neutral          | age           |   0.258 |    0.161 |     0.346 |
| talk_neutral          | law           |   0.244 |    0.175 |     0.315 |
| talk_neutral          | social_status |  -0.147 |   -0.231 |    -0.055 |
| talk_neutral          | intervention  |  -0.095 |   -0.152 |    -0.047 |
| persona_neutral       | intervention  |   0.022 |    0.007 |     0.04  |

![persuasion_heatmap.png](persuasion_heatmap.png)

## 4. Emotions inside the model

![probes.png](probes.png)

## 5. Steering

Mean |Δ AMCE| over the 9 dimensions (an emotion method should beat its random control):

|                 |   direction |   neurons |   random |   random_neurons |
|:----------------|------------:|----------:|---------:|-----------------:|
| ('anger', 0.5)  |       0.145 |     0.115 |    0.101 |            0.141 |
| ('anger', 0.75) |       0.12  |     0.073 |    0.138 |            0.167 |
| ('fear', 0.5)   |       0.149 |     0.115 |    0.062 |            0.148 |
| ('fear', 0.75)  |       0.145 |     0.132 |    0.139 |            0.135 |
| ('hope', 0.5)   |       0.137 |     0.151 |    0.128 |            0.168 |
| ('hope', 0.75)  |       0.084 |     0.135 |    0.102 |            0.156 |

![steering_heatmap.png](steering_heatmap.png)

![dose_fear.png](dose_fear.png)

![dose_anger.png](dose_anger.png)

![dose_hope.png](dose_hope.png)

## 6. Side effects

![side_effects.png](side_effects.png)

## 7. Time and energy

| step                             | model                    |   n_forward |   seconds |   sec_per_forward |         kwh |   wh_per_1000_forward |
|:---------------------------------|:-------------------------|------------:|----------:|------------------:|------------:|----------------------:|
| baseline                         | Qwen/Qwen2.5-7B-Instruct |        1800 |    355.37 |            0.1974 | 0.00776369  |                4.3132 |
| comprehension                    | Qwen/Qwen2.5-7B-Instruct |        1200 |    278.45 |            0.232  | 0.00606772  |                5.0564 |
| baseline                         | Qwen/Qwen2.5-7B-Instruct |        3000 |    601.67 |            0.2006 | 0.013084    |                4.3613 |
| baseline                         | Qwen/Qwen2.5-7B-Instruct |        3600 |    777.69 |            0.216  | 0.0169127   |                4.698  |
| comprehension                    | Qwen/Qwen2.5-7B-Instruct |        1200 |    260.88 |            0.2174 | 0.00567802  |                4.7317 |
| probe_activations                | Qwen/Qwen2.5-7B-Instruct |         320 |     11.18 |            0.0349 | 0.000230063 |                0.7189 |
| steering                         | Qwen/Qwen2.5-7B-Instruct |        9000 |   2695.17 |            0.2995 | 0.0590558   |                6.5618 |
| probe_activations                | Qwen/Qwen2.5-7B-Instruct |         320 |     12.02 |            0.0376 | 0.000247407 |                0.7731 |
| steering                         | Qwen/Qwen2.5-7B-Instruct |        9000 |   2730.52 |            0.3034 | 0.0598582   |                6.6509 |
| persuasion:baseline              | Qwen/Qwen2.5-7B-Instruct |         480 |     74.62 |            0.1555 | 0.0016631   |                3.4648 |
| persuasion:talk_fear             | Qwen/Qwen2.5-7B-Instruct |         480 |     99.9  |            0.2081 | 0.00221683  |                4.6184 |
| persuasion:talk_anger            | Qwen/Qwen2.5-7B-Instruct |         480 |     96.98 |            0.202  | 0.00214873  |                4.4765 |
| persuasion:talk_hope             | Qwen/Qwen2.5-7B-Instruct |         480 |     96.76 |            0.2016 | 0.00214813  |                4.4753 |
| persuasion:persona_utilitarian   | Qwen/Qwen2.5-7B-Instruct |         480 |     84.16 |            0.1753 | 0.00186749  |                3.8906 |
| persuasion:persona_deontologist  | Qwen/Qwen2.5-7B-Instruct |         480 |     85.43 |            0.178  | 0.00189429  |                3.9464 |
| persuasion:persona_care_ethics   | Qwen/Qwen2.5-7B-Instruct |         480 |     84.45 |            0.1759 | 0.00187248  |                3.901  |
| persuasion:country_France        | Qwen/Qwen2.5-7B-Instruct |         480 |     82.9  |            0.1727 | 0.00183582  |                3.8246 |
| persuasion:country_Japan         | Qwen/Qwen2.5-7B-Instruct |         480 |     83.08 |            0.1731 | 0.00183722  |                3.8275 |
| persuasion:country_United States | Qwen/Qwen2.5-7B-Instruct |         480 |     83.16 |            0.1732 | 0.00184501  |                3.8438 |
| persuasion:country_Brazil        | Qwen/Qwen2.5-7B-Instruct |         480 |     83.15 |            0.1732 | 0.00184258  |                3.8387 |
| probe_activations                | Qwen/Qwen2.5-7B-Instruct |         320 |     12.87 |            0.0402 | 0.000260264 |                0.8133 |
| probe_activations                | Qwen/Qwen2.5-7B-Instruct |         320 |     11.78 |            0.0368 | 0.000241659 |                0.7552 |
| persuasion:talk_neutral          | Qwen/Qwen2.5-7B-Instruct |         480 |     92.25 |            0.1922 | 0.00205018  |                4.2712 |
| persuasion:persona_neutral       | Qwen/Qwen2.5-7B-Instruct |         480 |     78.44 |            0.1634 | 0.00173735  |                3.6195 |
