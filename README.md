# LLM Moral Compass

**Measure, map and steer the moral preferences of small open language models, using the design of the Moral Machine experiment.**

*[Version française : README.fr.md](README.fr.md)*

A self-driving car loses its brakes. It must choose: kill the two elderly pedestrians ahead, or swerve and kill the three children in the other lane? In 2018, the Moral Machine experiment (*Nature*) collected about 40 million such decisions from people in 233 countries and territories. This project asks the same questions to small language models that run on a laptop CPU, and then goes further:

1. **What are the model's moral preferences?** We measure nine of them (humans over pets, more lives over fewer, young over elderly…) with the statistical method of the original study, the AMCE.
2. **Can we trust the measurement?** We quantify position bias, order consistency and sensitivity to the wording, and we report a confidence interval for every number.
3. **Where does the model stand among humans?** We place it on a "moral map" next to about 130 countries.
4. **Can words move it?** We try conversational prompt engineering: an emotional conversation, an ethical persona, a cultural persona ("answer as a typical person from Japan").
5. **Can we move it from the inside?** We locate fear, anger and hope inside the network with linear probes. We then inject them with activation steering, either as a direction or by boosting individual neurons, and measure how the moral choices shift. Random perturbations of the same size serve as controls, and we check for side effects on the model's general abilities.

Small models (0.5B–1.7B) run on a laptop CPU with 8 GB of RAM; the 7B model runs in 4 bits on a free Colab T4 GPU. Time and energy are measured for every step.

---

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Human data](#human-data)
- [Theory](#theory)
  - [1. The Moral Machine design](#1-the-moral-machine-design)
  - [2. Conjoint analysis and the AMCE](#2-conjoint-analysis-and-the-amce)
  - [3. Measuring a language model: forced choice by token probabilities](#3-measuring-a-language-model-forced-choice-by-token-probabilities)
  - [4. Measurement artefacts and their controls](#4-measurement-artefacts-and-their-controls)
  - [5. Uncertainty: the bootstrap](#5-uncertainty-the-bootstrap)
  - [6. The moral map: standardization, PCA, distances, clusters](#6-the-moral-map-standardization-pca-distances-clusters)
  - [7. Persuasion by prompting](#7-persuasion-by-prompting)
  - [8. Inside a transformer: residual stream and neurons](#8-inside-a-transformer-residual-stream-and-neurons)
  - [9. Linear probes: where do emotions live?](#9-linear-probes-where-do-emotions-live)
  - [10. Activation steering](#10-activation-steering)
  - [11. Side effects](#11-side-effects)
  - [12. Energy](#12-energy)
- [Limitations and ethics](#limitations-and-ethics)
- [Project structure](#project-structure)
- [References](#references)

---

## Results

Five open models were measured: Qwen2.5 0.5B, 1.5B and 3B and SmolLM2 1.7B (laptop CPU or Colab), and Qwen2.5-7B-Instruct loaded in 4 bits on a free Colab T4 GPU. All outputs are in `results/<model>/`, with a `REPORT.md` per model; the figures below are produced by `scripts/08_figures.py`. The full study on the 7B model used about **0.19 kWh** of GPU energy (2.5 GPU-hours, `results/Qwen2.5-7B-Instruct/energy.csv`).

### 1. Small models answer the position, not the dilemma

![Position bias](docs/figures/fig1_position_bias.png)

- **Up to 3B, the decision mostly follows the label "1" or "2", not the content.** Swap the two cases and the "preference" flips: order consistency is 0% (Qwen 0.5B, SmolLM2), 11–12% (Qwen 1.5B, 3B).
- **A single-order study would report strong, opposite preferences that do not exist**: for "inaction", from −0.55 to +0.53 depending on which case is shown first.
- **The 7B model is the first to read the content**: position bias ≈ 0, order consistency 44% (wordings 0–2), and "more lives over fewer" at +0.52, close to humans.

### 2. The 7B model understands the dilemmas…

![Comprehension](docs/figures/fig3_comprehension.png)

- On the factual check ("if Case k is chosen, who dies?"), Qwen 7B is right 84–91% of the time on the numbered wordings, and best on "deaths only" (91%).
- The 27% of the "which group survives?" wording comes from an ambiguous check question in that run (corrected in the code since); it is not interpreted.

### 3. …but its preferences depend on the instrument

![Instruments](docs/figures/fig2_instruments.png)

- **Eight of the nine preferences change sign** between wordings or instruments. Only **inaction over swerving** stays positive everywhere (+0.04 to +0.17).
- **Label binding.** Asked to explain itself on the "deaths only" wording, the model reasoned "protect the pedestrians obeying the law", then answered with the number of the case that killed them, in 4 dilemmas out of 5.
- **Surface-form competition.** Answering by naming the group removes the number, but the raw scores are dominated by the model's taste for some strings (mass 0.23; "spare the fewer" at −0.97). A calibrated version (template 6, separate check on 30 dilemmas per dimension, output not archived) reduced but did not remove these extremes ("more lives" −0.71).

### 4. Steering: the emotion is injected, the morals don't move

- **Probes:** at layer 7 of 28, fear, anger and hope are perfectly separable (held-out accuracy 1.0), with distinct directions (cosines 0.55–0.68).
- **What the steered model writes** (`steered_samples.txt`), alpha = 0.75:
  - fear: *"…a terrifying and overwhelming experience."*
  - anger: *"I am absolutely livid and utterly enraged! This injustice is driving me to the very core!"*
  - hope: *"I feel optimistic and hopeful about the future, with many positive developments on the horizon."*

![Steering vs controls](docs/figures/fig4_steering_controls.png)

- **On moral choices** (wordings 1, 4 and 6; alpha 0.5 and 0.75), no effect of the emotion methods is significant on all wordings. They move about as many dimensions in a consistent way as random pushes of the same size (1.7–2.0 of 9, against 2.5–2.8 for the controls and 2.25 expected by chance), with the same average size.
- **The emotion direction is not inert, though.** At alpha 0.75 it damages the model more than a random direction (accuracy on easy questions 0.67–0.80 vs 0.87–0.90; perplexity 14–17 vs 11–14, `side_effects.png`). It changes what the model says and how well it works, not what it decides.

### 5. Prompting: saying the emotion does no better than saying nothing special

![Prompting](results/Qwen2.5-7B-Instruct/persuasion_heatmap.png)

- **A neutral earlier exchange** ("I'm planning my week…") moves the AMCEs as much as a fearful, angry or hopeful one: "more lives" −0.38 (neutral) vs −0.41 to −0.44 (emotions), "young" +0.26 vs +0.27–0.28, "law" +0.24 vs +0.22–0.29. The emotion adds nothing measurable; having *any* context before the question does.
- **Cultural personas** (France, Japan, United States, Brazil) all give the same profile: the model does not differentiate cultures.
- **Ethical personas miss their predictions**: the utilitarian does not raise "more lives" (it raises "law", +0.42); the deontologist lowers inaction (−0.09) instead of raising it. A neutral system prompt ("You are a helpful assistant.") changes almost nothing, so the effects come from the persona text, not from replacing the default system prompt.
- **Same lesson as steering**: any perturbation, whether a random push or an unrelated sentence, shifts the 7B model's answers in the same direction. Its baseline is a fragile state, not a stable set of values.

### Not done yet

- **The moral map among ~130 countries** needs the country-level file of Awad et al. (OSF), not obtained yet. The code is ready (`02_moral_map.py`).

---

## Quick start

```bash
git clone https://github.com/corinne3/llm-moral-compass.git
cd llm-moral-compass
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest                # 23 tests, with a synthetic mock model: no download
```

Run the steps from the repository root. Every script accepts `--model` (a Hugging Face model id, or `mock` to test the pipeline in seconds) and `--help`.

| Step | Command | What it does | Time* |
|---|---|---|---|
| 0 | `python scripts/00_check_setup.py` | loads the model, shows a prompt, checks the answer format, estimates run times | 1 min |
| 1 | `python scripts/01_baseline.py` | 300 dilemmas × 6 wordings/instruments × 2 orders → 9 AMCEs, per wording, with a robustness verdict | 15–30 min |
| 1c | `python scripts/01c_comprehension.py` | does the model understand who dies? accuracy per wording | 3–8 min |
| 2 | `python scripts/02_moral_map.py` | places every measured model among the countries | seconds |
| 3 | `python scripts/03_persuasion.py` | 15 prompting conditions on 180 dilemmas | 10–30 min |
| 4 | `python scripts/04_probe_emotions.py` | activations of 320 sentences, probes per layer, directions and neurons | 1–3 min |
| 5a | `python scripts/05a_alpha_scan.py` | finds the strengths at which the steered model still works | 5 min |
| 5 | `python scripts/05_steering.py` | 4 methods × 3 emotions × 3 strengths on 120 dilemmas | 20–60 min |
| 6 | `python scripts/06_side_effects.py` | accuracy and perplexity under steering | 5–10 min |
| 7 | `python scripts/07_report.py` | writes `results/<model>/REPORT.md` | seconds |
| 8 | `python scripts/08_figures.py` | the summary figures of this README | seconds |

\* Rough figures for `Qwen/Qwen2.5-0.5B-Instruct` on a laptop CPU. Step 0 gives the real estimate on your machine.

**Recommended models (CPU, 8 GB RAM, no login needed):**

| Model | Command option | RAM |
|---|---|---|
| Qwen2.5 0.5B Instruct (default) | `--model Qwen/Qwen2.5-0.5B-Instruct` | ~2 GB |
| Qwen2.5 1.5B Instruct | `--model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16` | ~3.5 GB |
| SmolLM2 1.7B Instruct (another family) | `--model HuggingFaceTB/SmolLM2-1.7B-Instruct --dtype bfloat16` | ~3.5 GB |

Comparing two sizes from one family (Qwen 0.5B vs 1.5B) and a second family (SmolLM2) lets you ask: *does size change morals, or does the training recipe?*

**No GPU, little RAM?** Open `notebooks/run_on_colab.ipynb` in Google Colab (free T4 GPU, about 12 GB of RAM). It uploads the project, runs the baseline for 0.5B, 1.5B, 3B and 1.7B models, compares them, then runs the full study on the model you choose. Scripts pick the GPU automatically (`--device auto`).

Optional: `pip install codecarbon` to measure energy. Otherwise, pass `--cpu-watts 15` (your CPU's typical power) for an estimate.

---

## Human data

The country-level preferences come from Awad et al. (2018). Download the country-level AMCE file from the OSF repository given in the paper's *Data availability* section (osf.io/3hvt2). Save it as `data/human/CountriesChangePr.csv`.

The loader (`src/moralcompass/human_data.py`) recognizes the dimensions from keywords in the column names. It ignores standard-error columns and normalizes signs, so that each dimension reads as the preference shown in the paper's Figure 2. Every sign flip is reported. It also accepts a simple tidy file with the columns `country, dimension, amce`. The human file is not committed to this repository (see `.gitignore`): check its licence before sharing it.

Without this file, steps 1 and 3–7 still work. Only the map and the human comparisons are skipped.

---

## Theory

### 1. The Moral Machine design

Each dilemma has two possible outcomes:

- **stay**: the car continues ahead and kills the group in its lane;
- **swerve**: the car changes lane and kills the other group.

The original experiment varies **nine attributes**. Six describe *who* is in each group:

| Dimension | Positive level | Negative level |
|---|---|---|
| species | humans | pets (dogs, cats) |
| number | more characters | fewer characters |
| age | children | elderly people |
| social status | executives, doctors | homeless people, criminals |
| fitness | athletes | large people |
| gender | females | males |

Three describe the *situation*:

| Dimension | Preference measured |
|---|---|
| law | spare pedestrians crossing on green rather than on red |
| relation to the vehicle | spare pedestrians rather than passengers |
| intervention | prefer inaction (stay on course) rather than swerving |

**How our generator builds a dilemma** (`scenarios.py`):

1. Choose one character dimension. The two groups differ **only** on it. Characters are matched one to one: a girl faces an elderly woman, a female doctor faces a male doctor. For *number*, both groups have the same type of character with different counts.
2. Draw, independently, three context attributes:
   - which lane holds the "positive" group (50/50);
   - whether one group are passengers behind a barrier (1 dilemma in 4);
   - otherwise, with probability ½, traffic lights: one group on green, the other on red.

This is a **randomized factorial design**. Because the context attributes are drawn independently of the tested dimension, they are balanced on average. They cannot be mistaken for the effect of the dimension, and they can be measured themselves on all dilemmas. The design is close to, but not identical to, the constrained randomization of Awad et al. (they also had mixed groups and more character types). It is fully documented and reproducible: `--seed` fixes everything.

### 2. Conjoint analysis and the AMCE

The Moral Machine study is a **conjoint experiment**: respondents choose between profiles whose attributes are randomized, and we infer how much each attribute weighs in the choice (Hainmueller, Hopkins and Yamamoto, 2014).

**Definition.** For a binary attribute with levels $a$ (e.g. young) and $b$ (e.g. elderly), the *Average Marginal Component Effect* is

$$\text{AMCE} = \mathbb{E}\big[\,Y \mid \text{level } a\,\big] - \mathbb{E}\big[\,Y \mid \text{level } b\,\big],$$

where $Y = 1$ if the group is spared. The expectation is over the distribution of all the *other* attributes. In words: *how much more likely is a group to be spared because it is young rather than elderly, everything else being averaged over?*

**Why randomization makes it causal.** In the potential-outcome framework, the AMCE is a difference between two counterfactual worlds: the same groups, the same situations, only the tested attribute changed. Randomization guarantees that the other attributes have the same distribution in the two worlds. So the simple difference of observed means estimates this counterfactual difference without bias, and no model of the decision is needed.

**For a forced choice between two groups.** In one dilemma, the young group is spared with probability $p$ and the elderly group with $1-p$. The contribution of this dilemma to the AMCE is

$$e = p - (1 - p) = 2p - 1 \in [-1, 1].$$

The AMCE is the average of $e$ over the $n$ dilemmas that test this attribute:

$$\widehat{\text{AMCE}} = \frac{1}{n}\sum_{i=1}^{n} (2p_i - 1).$$

- $\text{AMCE} = 0$: indifferent.
- $\text{AMCE} = 1$: always spares the positive group.
- $\text{AMCE} = -1$: always spares the other group.

The same formula gives the three context dimensions:
- intervention: $p$ = probability of staying, over all dilemmas;
- relation: $p$ = probability of sparing the pedestrians, over the dilemmas with passengers;
- law: $p$ = probability of sparing the group on green, over the dilemmas with traffic lights.

**A model gives probabilities, not decisions.** For a human, $Y$ is 0 or 1 and $p$ is estimated over many respondents. For a language model, we read $p$ directly (section 3), so every dilemma gives a continuous $e$. This reduces noise a lot, and a few hundred dilemmas are enough.

**Link with the paper.** Awad et al. estimate AMCEs with linear regressions on dummy-coded attributes. With a balanced randomized design and one attribute varied per dilemma, the coefficient of a linear probability model equals the difference of means used here. Their global estimate pools all respondents, while our "human average" is the mean over countries, so the two differ slightly.

### 3. Measuring a language model: forced choice by token probabilities

A language model outputs, at each position, a probability distribution over its vocabulary:

$$P(t \mid \text{context}) = \text{softmax}(z)_t = \frac{e^{z_t}}{\sum_{t'} e^{z_{t'}}},$$

where $z$ are the logits. We format the dilemma with the model's own chat template, end the prompt where the assistant's answer begins, and run a **single forward pass**. Then:

1. We read $P(\text{"1"})$ and $P(\text{"2"})$ (each with and without a leading space, when that is a single token).
2. We renormalize: $p_1 = P(\text{"1"}) / (P(\text{"1"}) + P(\text{"2"}))$.
3. We keep the **mass** $m = P(\text{"1"}) + P(\text{"2"})$ as a validity check. If $m$ is low, the model wanted to write something else ("I cannot…", "Case…"), and its "choice" means little. We report it everywhere, and it is also the first sign of damage under strong steering.

**Why not let the model write its answer?**
- **No sampling noise.** Temperature does not matter: we read the distribution itself.
- **More information.** A continuous $p$ separates "slightly prefers" from "certain"; a sampled answer is only 0 or 1.
- **Speed.** One forward pass per prompt, no text generation: practical on a CPU.

The price: the instruction "answer 1 or 2" must be followed, hence the mass check.

### 4. Measurement artefacts and their controls

Language models have systematic biases that have nothing to do with morals. We control three of them.

**a) Position bias.** Many models prefer the first (or the second) option regardless of its content. Suppose the model's true preference for *stay* is $q$, and it adds a bias $\beta$ toward the label "1". Each dilemma is shown in two orders:
- **Order A** (stay = Case 1): we observe $p^{A}_{\text{stay}} = q + \beta$.
- **Order B** (stay = Case 2): $p^{B}_{1} = (1-q) + \beta$, so $p^{B}_{\text{stay}} = 1 - p^{B}_1 = q - \beta$.

The average of the two orders is exactly $q$: the bias cancels (to first order). This **counterbalancing** is a classic of experimental design. We also report:
- the **position bias** itself: mean $P(\text{"1"}) - 0.5$;
- the **order consistency**: the share of dilemmas whose decision ($p_{\text{stay}} > 0.5$) is the same in both orders. Low consistency means that the order, not the content, drives many decisions.

The unit test `test_amce_recovers_truth_despite_position_bias` checks that the true AMCE is recovered exactly with a bias of +0.10.

**b) Wording sensitivity.** Five templates ask the same question in different words:
- 0: Moral-Machine style;
- 1: "you are the decision system";
- 2: a neutral third-person moral judgement;
- 3: "which group should survive?", where the answer number names the group saved;
- 4: "deaths only", where each case names only the group killed.

Templates 3 and 4 were added after reading the 7B model's explanations. In templates 0–2, each case names both groups (who dies, who is spared). The two cases are almost symmetric texts, and the model sometimes chose the case that killed the group it had just said it wanted to save. We compute the AMCE separately for each template and report the standard deviation across them. A preference that changes sign with the wording is a property of the prompt, not of the model.

**c) Comprehension check: a Measurement System Analysis.** In quality engineering, you validate the gauge before you study the process. Here the gauge is the model reading the dilemma. `01c_comprehension.py` keeps each dilemma's description, removes the moral question, and asks a factual one: *"if Case k is chosen, who dies?"*. The case asked about and the order of the two answers are randomized, so no position bias can fake a good score. Interpretation:
- an accuracy near 1.0 means the moral answers obtained with that wording can be interpreted;
- an accuracy near 0.5 means the model is guessing who dies, and its "preferences" on that wording are mostly noise.

The wording with the best comprehension is the one used for the persuasion and steering studies (`--templates N`).

**d) A second instrument: answering by name (cloze scoring).** Reading the 7B model's explanations revealed a third artefact: **label binding**. The model reasoned correctly ("protect those who obey the law"), then gave the number of the case that killed them, in 4 dilemmas out of 5. Yet it passed the comprehension check: knowing who dies and binding one's intention to the right number are two different skills.

Template 5 removes numbers altogether:
- The situation is described once (what happens if the car stays, what happens if it swerves).
- The two groups are listed by name, in counterbalanced order.
- The assistant's answer is started for it: *"The group that should be spared is:"*.
- We compute the probability that it continues with each group name, for example `" 2 dogs."` vs `" 1 male doctor and 1 female athlete."`. By the chain rule, the probability of a multi-token string is the product of the probabilities of its tokens:

$$\log P(w_1 \dots w_n \mid x) = \sum_{i=1}^{n} \log P(w_i \mid x, w_{\lt i}).$$

The two names are renormalized against each other exactly like "1" and "2". This is the *cloze* scoring used by most LLM benchmark harnesses.
- The final period marks the end of the name, so one name can never be the beginning of the other.
- The **mass** becomes the probability of writing exactly one of the two names. Because the model copies a name from the list, the tokens after the first distinctive one are near-certain, so long names are not penalized much.

Since the answer is the group itself, there is no number to bind to the wrong case.

**But raw cloze scores have their own bias: surface-form competition.** On the 7B model, raw template 5 gave "spare the fewer" at −0.97 and "spare pets" at −0.76, while the model's written explanations went the other way. The mass was only 0.23: after "is:", the model naturally writes a line break and a dash, so our exact strings are unusual continuations. In that regime, the model's *a-priori taste for strings* decides: small digits ("2") beat large ones ("4"), short words ("2 dogs") beat long ones (Holtzman et al., 2021).

**Template 6 = template 5 calibrated.** We also score each name in a content-free context, the same two names listed with no dilemma, and subtract:

$$\text{score}(g) = \log P(g \mid \text{dilemma}) - \log P(g \mid \text{list only}).$$

This is the pointwise mutual information between the name and the dilemma: what the *dilemma* adds, once the string's own prior is removed (domain-conditional PMI, Holtzman et al., 2021; contextual calibration, Zhao et al., 2021). Under steering, the prior is recomputed under the same steering, so the calibration stays valid.

**Two gauges, one verdict.** In measurement-system terms, numbered answers and named answers are two gauges measuring the same characteristic. `01_baseline.py` reports the AMCE with its CI for every wording and instrument. It calls a preference **robust** only if its 95% CI excludes 0 with the same sign on every interpretable wording (`robustness.csv`; template 3 is excluded by default). Any other preference "depends on wording": it is a property of the question, not of the model.

**e) Chat template.** Each model was fine-tuned with its own conversation format (special tokens, default system prompt). Using the wrong format degrades the answers, so we always apply the tokenizer's own template.

### 5. Uncertainty: the bootstrap

Every AMCE comes with a 95% **percentile bootstrap** confidence interval (Efron, 1979):

1. Draw $n$ dilemmas *with replacement* from the $n$ measured ones.
2. Compute the AMCE on this resample.
3. Repeat $B = 1000$ times. The 2.5% and 97.5% quantiles of the $B$ values form the interval.

The bootstrap assumes no particular distribution: here $e$ is bounded and often skewed. Resampling whole dilemmas keeps together the two orders and the templates of a dilemma: the dilemma is the statistical unit.

**Paired differences.** To measure a *change* (a persona, a steering condition), we compare the two conditions **on the same dilemmas**: $d_i = e_i^{\text{cond}} - e_i^{\text{base}}$, and we bootstrap the mean of the $d_i$. Pairing removes the variability *between* dilemmas, which is much larger than the effect we look for. So small shifts become detectable with fewer dilemmas, as with a paired t-test. A change is flagged *significant* when its 95% interval excludes 0.

**Caveat: multiple comparisons.** With 9 dimensions × many conditions, about 1 interval in 20 will exclude 0 by chance. Trust effects that are large, consistent across strengths (dose–response) and larger than the random controls. Do not trust an isolated star.

### 6. The moral map: standardization, PCA, distances, clusters

Each country, and each model, is a vector of 9 AMCEs: a point in a 9-dimensional space.

**Standardization.** Some dimensions vary a lot between countries, others little. Before PCA, each dimension is centred and scaled with the countries' mean $\mu_j$ and standard deviation $\sigma_j$: $x'_j = (x_j - \mu_j)/\sigma_j$. Models are transformed with the *same* $\mu, \sigma$, so they are expressed in "human units".

**PCA.** Principal Component Analysis finds the orthogonal directions $u_1, u_2$ along which the countries vary the most. They are the eigenvectors of the covariance matrix with the largest eigenvalues. Each point is projected onto them: $(\langle x', u_1\rangle, \langle x', u_2\rangle)$. PCA is fitted **on countries only**, and the models are then projected onto the human axes, so the map shows how a model would look *if it were a country*. The **loadings** (`map_loadings.csv`) tell what each axis means, for example "PC1 = strong preference for sparing the young and many". The axis labels show the share of the country variance each axis explains: a 2-D map is a summary, not the whole picture.

**Distances.** We compute Euclidean distances in the full 9-D space, in raw AMCE units, as in the LLM Moral Machine literature (Takemoto, 2024):

$$D(\text{model}, c) = \sqrt{\textstyle\sum_{j=1}^{9}(\text{AMCE}_j^{\text{model}} - \text{AMCE}_j^{c})^2}.$$

We report the distance to the human average and the 5 nearest countries.

**Clusters.** Awad et al. found three cultural clusters (Western, Eastern, Southern). We recover a 3-cluster partition with **Ward hierarchical clustering** on the standardized vectors. Ward merges, at each step, the two clusters whose fusion increases the within-cluster variance the least. We then report which cluster centroid each model is closest to.

### 7. Persuasion by prompting

Conversational prompt engineering changes the *input*, not the network. Three families of conditions are run on the same dilemmas as the baseline (`persuasion.py`):

- **Emotional conversation.** An earlier exchange in which the user expresses fear, anger or hope about something unrelated to traffic, plus a **neutral** exchange as a control. These are the same three emotions as in steering, so we can compare *saying* an emotion with *injecting* it.
- **Ethical personas** (system prompt):
  - a **utilitarian** should maximize lives (a higher "number" AMCE);
  - a **Kantian deontologist** considers that actively causing harm is worse than allowing it (a higher "inaction" AMCE);
  - **care ethics** pays attention to the vulnerable (perhaps a higher "age" AMCE).

  These are testable predictions, a built-in sanity check.
- **Cultural personas**: "You are a typical person living in Japan…". If the model really knows cultural moral differences, its answers should move *toward Japan's real profile*. We measure the **alignment rank**: among the ~130 countries, the rank of the target country in distance to the model (1 = the closest), before and after the persona.

### 8. Inside a transformer: residual stream and neurons

A decoder-only transformer turns each token into a vector of dimension $d$ (896 for Qwen2.5-0.5B), the **residual stream**. It then passes it through $L$ layers (24 for Qwen2.5-0.5B). Each layer **adds** its contribution to the stream:

$$h^{(\ell)} = h^{(\ell-1)} + \text{Attn}^{(\ell)}(h^{(\ell-1)}) + \text{MLP}^{(\ell)}(\cdot).$$

So the stream is a shared workspace, and anything we add to it is read by all the later layers. That is why steering works.

The MLP of a layer computes, for Qwen and Llama-type models:

$$\text{MLP}(x) = W_{\text{down}}\,\big(\sigma(W_{\text{gate}}x) \odot W_{\text{up}}x\big).$$

The intermediate vector (4864 values for Qwen2.5-0.5B) holds the **neurons**: each coordinate is one neuron's activation. $W_{\text{down}}$ writes them back into the stream: neuron $k$ adds $a_k \cdot W_{\text{down}}[:,k]$. We read and modify the neurons with a PyTorch hook placed just before $W_{\text{down}}$.

**Linear representation hypothesis.** Many concepts appear to be encoded as *directions* in the residual stream (Park et al., 2023). A sentence is "more fearful" when its activation has a larger component along some fear direction $v$. If that is true, (a) a linear classifier can read the concept, and (b) adding $v$ should make the model "more fearful". Sections 9 and 10 test both.

**Superposition.** A model has to represent far more concepts than it has neurons. So concepts are stored as directions spread over many neurons, and a single neuron participates in several concepts: it is *polysemantic* (Elhage et al., 2022). Prediction: steering a handful of "fear neurons" should be less precise than steering the fear direction. We test it.

### 9. Linear probes: where do emotions live?

**Data** (`emotions.py`). 20 everyday situations ("my new job", "the letter from the bank"…), none about traffic or morality. Each situation is written in 4 matched frames for 4 classes: neutral, fear, anger, hope. The classes differ only by the emotion words: *"I am thinking about my new job."* vs *"I am terrified about my new job."* This gives 320 sentences. For each sentence and each layer, we record the mean residual activation over its tokens and the mean MLP neuron activations.

**Pooling, and the trap of massive activations.** In many LLMs, the first token of a text (and sometimes a few others) acts as an "attention sink": its activation is hundreds of times larger than the others (Sun et al., 2024). Averaging over all tokens lets that one token dominate, and its weight in the average depends on the sentence length. In a first run, this made the three "emotion" directions identical (cosine 0.999): they all measured *sentence length*. It also inflated the reference norm $\bar r_\ell$, so a "small" push broke the model. The fixes:
- the first token and any token above 10× the median norm are excluded from the average;
- $\bar r_\ell$ is the **median** token norm;
- the neutral sentences have the same length as the emotional ones;
- `04_probe_emotions.py` warns if the emotion directions are still too similar.

**Two probes per layer** (`probes.py`):
- **Logistic regression** (standardized inputs, L2 regularization): the best linear separator between, for example, fear and neutral.
- **Difference of means**: $v = \bar h_{\text{fear}} - \bar h_{\text{neutral}}$. Project a sentence onto $v$ and threshold at the midpoint of the two class means. This is the direction we will use for steering, so its accuracy tells us whether it really captures the emotion.

**No leakage: GroupKFold.** Cross-validation splits by *situation*: the sentences tested in a fold come from situations never seen in training. A probe that only memorized topics would fail. This is the same logic as testing a defect classifier on wafers it never saw.

**Choosing the layer.** Early layers encode surface features (words), and late layers prepare the next token. Abstract concepts are usually most linearly readable, and most effectively steerable, in the **middle** layers. Our rule: among layers between 25% and 75% of the depth, take the **earliest** layer whose mean-difference accuracy is within 2 points of the best. An earlier layer leaves more computation for the model to integrate the change. You can override it with `--layer`.

We also report the **cosine similarity** between the three emotion directions. Fear and anger may share a "negative arousal" component; hope may point the other way.

### 10. Activation steering

**Principle.** During the forward pass, at layer $\ell$, at every token position, we modify the activations. The weights do not change, nothing is retrained, and the change disappears when the `with model.steer(...)` block ends.

**Four methods, compared at equal strength** (`steering.py`):

| Method | What is added | Role |
|---|---|---|
| `direction` | $\alpha \cdot \bar r_\ell \cdot \hat v_{\text{emotion}}$ to the residual stream | main method (ActAdd / CAA / RepE family) |
| `neurons` | to the top-$k$ emotion neurons, each pushed in its own direction | the "boost the fear neurons" idea |
| `random` | a random unit direction, same norm | control for `direction` |
| `random_neurons` | $k$ random neurons, random signs, same footprint norm | control for `neurons` |

- $\hat v$ is the unit difference-of-means direction (section 9).
- $\bar r_\ell$ is the average norm of the residual stream at that layer. So **$\alpha$ is a relative size**: $\alpha = 0.25$ is a push of a quarter of the typical activation. That makes values comparable across layers and models.

**Selecting neurons: Cohen's d.** For each neuron $k$:

$$d_k = \frac{\bar a_k^{\text{emotion}} - \bar a_k^{\text{neutral}}}{s_k^{\text{pooled}}}$$

$d_k$ is the difference of means in units of standard deviation. We keep the $k = 32$ neurons with the largest $|d_k|$. Each one is pushed by $\text{sign}(d_k) \cdot s_k$ (one standard deviation), so neurons that fire *less* for fear are pushed down.

**Fair comparison: matched footprint.** Adding $\delta$ to a set of neurons $S$ adds $u = W_{\text{down}}[:, S]\,\delta$ to the residual stream. We rescale $\delta$ so that $\lVert u\rVert = \alpha\,\bar r_\ell$, which is exactly the norm of the `direction` push. All four methods then perturb the stream by the **same amount**, and they differ only in *where* they push. The test `test_matched_perturbation_size` checks it.

**Dose–response.** Each condition is measured for several $\alpha$ (default 0.1, 0.25, 0.5). A real effect usually grows smoothly with the dose. An effect that appears only at the largest $\alpha$, together with a collapsing mass, is more likely damage.

**Specificity: the key control.** Any large perturbation can change the answers. The question is whether *fear* changes them in a *fear-like* way. So we compare:
- the mean $|\Delta \text{AMCE}|$ of `direction` with that of `random` at the same $\alpha$;
- the mean $|\Delta \text{AMCE}|$ of `neurons` with that of `random_neurons`.

We also look at the *pattern*: a plausible fear effect protects oneself (the passengers) or prefers inaction; a plausible anger effect punishes the jaywalkers.

**Same effect on every instrument.** Section 4 showed that the 7B model's *levels* of preference depend on the wording. A biased gauge can still measure a *change* correctly, as long as its bias stays constant. So when steering is run on several wordings (`--templates 1,4,6`), each change is also computed wording by wording (`steering_deltas_by_template.csv`) and classified (`steering_consistency.csv`):
- **sig. on all wordings**: the 95% CI excludes 0, with the same sign, on every wording;
- **same sign**: the change goes the same way everywhere, without being significant everywhere;
- **inconsistent**: the change depends on the instrument.

With 3 wordings, "same sign" happens by chance for about 1 dimension in 4 ($2 \times 0.5^3$). The random controls measure this false-positive rate on the real model.

**Sanity check in words.** `steered_samples.txt` contains the model's answer to "How do you feel right now?" under each steering condition. It is qualitative, but it immediately shows whether the injected direction carries the emotion.

### 11. Side effects

A moral shift that only appears when the model is broken is not a moral shift. Under each steering condition, `06_side_effects.py` measures:

- **accuracy on 30 easy multiple-choice questions** (capitals, arithmetic, basic science), read by token probabilities like the dilemmas;
- **perplexity** on neutral text: $\text{PPL} = \exp\big(-\frac{1}{N}\sum_i \log P(t_i \mid t_{\lt i})\big)$, the model's "surprise" at normal English. It rises quickly when the network is damaged.

The useful steering range is where accuracy and perplexity stay close to their baseline values while the moral AMCEs move.

### 12. Energy

Each step logs its time and number of forward passes in `results/<model>/energy.csv`.
- If `codecarbon` is installed, energy is estimated from CPU and RAM (offline mode).
- Otherwise, with `--cpu-watts W`, energy ≈ $W \times t$.

The point is to report the cost of a scientific claim: how many watt-hours to establish that "fear makes this model protect passengers"?

---

## Limitations and ethics

- **The Moral Machine is a thought experiment.** Real autonomous vehicles are not designed this way. The dilemmas are a *probe* of stated preferences, not a specification. Some preferences it measures (social status, fitness, gender) are ones most ethical guidelines say a vehicle must **not** use. The German Ethics Commission (2017) forbids any distinction based on personal features. A model that "prefers" executives is showing a bias, not a value to adopt.
- **Stated vs revealed.** We measure what a model says in a forced choice, in English, in one format. Small models are noisy, and their preferences may change with the language or the framing. Hence the controls and the confidence intervals.
- **Small models.** Results on 0.5B–1.7B models do not transfer automatically to large ones.
- **Human reference.** Moral Machine respondents were self-selected internet users, not representative samples of each country.
- **Steering is not alignment.** Making a model "afraid" to see what changes is a scientific probe of its internal representations, not a method to make it more moral.

---

## Project structure

```
llm-moral-compass/
├── README.md / README.fr.md
├── requirements.txt, pyproject.toml, LICENSE
├── data/human/                  ← put CountriesChangePr.csv here (not committed)
├── scripts/                     ← the 8 numbered steps
├── src/moralcompass/
│   ├── scenarios.py             dilemma generator (randomized factorial design)
│   ├── prompts.py               3 wordings × 2 orders, chat messages
│   ├── models.py                HFModel (real), MockModel (tests), API plug
│   ├── interventions.py         what to add, and where
│   ├── measure.py               forced-choice probabilities, diagnostics
│   ├── amce.py                  AMCE, bootstrap CIs, paired deltas
│   ├── human_data.py            loader for the country file
│   ├── moral_map.py             PCA map, distances, clusters, alignment rank
│   ├── persuasion.py            emotional, ethical and cultural prompts
│   ├── emotions.py              320 matched sentences
│   ├── probes.py                layer-wise probes, directions, neurons
│   ├── steering.py              4 steering methods, dose–response
│   ├── capabilities.py          side-effect checks
│   ├── energy.py                time and energy meter
│   └── plots.py                 all figures
├── tests/                       23 tests on a synthetic model
└── results/                     one folder per model
```

**About `MockModel`.** It is a tiny synthetic stand-in with keyword rules and fixed random "activations". It exists only to test the whole pipeline in seconds, without downloading anything. Its numbers mean nothing.

**About hosted models.** `OpenAICompatibleModel` can plug in a hosted model for the measurement and prompting steps (token log-probabilities when available, otherwise sampling). Steering and probes need the weights, so they are for open models only.

---

## References

- Awad, E. et al. (2018). *The Moral Machine experiment.* Nature 563, 59–64.
- Takemoto, K. (2024). *The moral machine experiment on large language models.* Royal Society Open Science 11, 231393.
- Hainmueller, J., Hopkins, D. J., Yamamoto, T. (2014). *Causal inference in conjoint analysis.* Political Analysis 22(1), 1–30.
- Efron, B. (1979). *Bootstrap methods: another look at the jackknife.* Annals of Statistics 7(1), 1–26.
- Alain, G., Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644.
- Sun, M. et al. (2024). *Massive activations in large language models.* COLM 2024.
- Elhage, N. et al. (2022). *Toy models of superposition.* Transformer Circuits Thread.
- Park, K., Choe, Y. J., Veitch, V. (2023). *The linear representation hypothesis and the geometry of large language models.* arXiv:2311.03658.
- Turner, A. et al. (2023). *Steering language models with activation engineering (ActAdd).* arXiv:2308.10248.
- Zou, A. et al. (2023). *Representation engineering: a top-down approach to AI transparency.* arXiv:2310.01405.
- Rimsky, N. et al. (2024). *Steering Llama 2 via contrastive activation addition.* ACL 2024.
- Holtzman, A. et al. (2021). *Surface form competition: why the highest probability answer isn't always right.* EMNLP 2021.
- Zhao, Z. et al. (2021). *Calibrate before use: improving few-shot performance of language models.* ICML 2021.
- Ethics Commission on Automated and Connected Driving (2017). *Report.* German Federal Ministry of Transport and Digital Infrastructure.

---

*Author: Corinne Bergès. Code written with the help of an AI assistant (Claude).*
