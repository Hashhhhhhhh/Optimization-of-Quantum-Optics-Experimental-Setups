# Optimization-of-Quantum-Optics-Experimental-Setups

# Photonic Graph Optimization — README

**Short description.**
This repository contains three Colab/Notebook implementations that demonstrate a *hybrid discrete+continuous* optimization pipeline for designing small photonic experiments using graph representations. The discrete structure (which pair sources / beam splitters to include) is chosen via a **QUBO** formulation and a classical annealer; the continuous knobs (pump amplitudes, source/mode phases, and beam-splitter angles) are tuned with **gradient-based optimization (autodiff)**. The three notebooks are:

* `notebooks/triangle_photonic.ipynb` — Triangle (3-mode) SPDC network → W/GHZ-like post-selected states.
* `notebooks/4mode_cluster.ipynb` — Linear 4-mode cluster / small cluster-state experiment (graph state target; Gaussian / perfect-matching variants included).
* `notebooks/star_repeater.ipynb` — Star-shaped repeater node (4 leaves); three distinct target states realized by different skeletons + continuous tuning.

These notebooks are suitable for Google Colab (GPU recommended) and intended for demonstration / reproducibility for a research group.

---

## Contents of this README

1. Overview of the optimization pipeline
2. Precise experiment descriptions (initial & target states) for each notebook
3. QUBO formulation — general recipe and specific choices used in the notebooks
4. Continuous optimization — objective, gradients, parameterizations, and practical choices
5. Implementation details (how to run, packages, outputs saved)
6. Validation, caveats, and recommended next steps for lab readiness
7. Reproducibility notes, files produced, and contact/license

---

## 1. Overview of the optimization pipeline

We decompose the design task into two nested problems:

1. **Discrete design (structure / skeleton selection).**
   Decide which components (SPDC sources, beam splitters, heralds, etc.) are present. Represent the discrete decisions as binary variables and build a QUBO $x^T Q x$ to encode rewards, synergies and constraints. Solve this QUBO with a classical annealer (simulated annealing provided) to get a small set of candidate skeletons.

2. **Continuous tuning (device knobs).**
   For each candidate skeleton, optimize continuous parameters (pump amplitudes $P$, source phases $\varphi$, local mode phases $\phi$, BS angles $\theta$) to maximize an objective—typically **fidelity** to a target state or a multi-objective combination of fidelity and expected count rate. Use autodiff (PyTorch) with Adam / L-BFGS for fast local refinement.

This modular pipeline lets the discrete search reduce combinatorial complexity while continuous optimization finds high-quality physical settings for each skeleton.

---

## 2. Experiments (initial / target states)

### A. Triangle photonic experiment (notebook: `triangle_photonic.ipynb`)

**Physical setup (initial):**

* 3 spatial modes (A,B,C).
* Up to three SPDC pair-sources: AB, AC, BC. Optionally local phase shifters on each mode, and one beam splitter between A–B (or multiple BSs in variants).
* Single-pair (low pump) approximation: the two-photon subspace is spanned by $|110\rangle$, $|101\rangle$, $|011\rangle$ (where e.g. $|110\rangle = a^\dagger b^\dagger|0\rangle$).

**Target state (example):** W-like state in the two-photon subspace

$$|\psi_{\text{target}}\rangle = \frac{1}{\sqrt{3}}\big(|110\rangle + |101\rangle + |011\rangle\big)$$

**How output is computed (single-pair algebra):**

* Source amplitudes: $c_{ij} = \kappa_{ij}\sqrt{P_{ij}} e^{i\varphi_{ij}}$
* Mode phases: $\tilde{c}_{ij} = c_{ij} e^{i(\phi_i+\phi_j)}$
* BS A–B with angle $\theta$: creation ops transform by $a^\dagger \mapsto \cos\theta \, a^\dagger + \sin\theta \, b^\dagger$, $b^\dagger \mapsto -\sin\theta \, a^\dagger + \cos\theta \, b^\dagger$
* Resulting coefficients (post-selected one-photon-per-mode basis):

$$C_{110} = \tilde{c}_{AB}\cos 2\theta, \quad C_{101} = \tilde{c}_{AC}\cos\theta - \tilde{c}_{BC}\sin\theta, \quad C_{011} = \tilde{c}_{AC}\sin\theta + \tilde{c}_{BC}\cos\theta$$

* Fidelity to target: 

$$F = \frac{|\sum_k C_k|^2}{3\sum_k|C_k|^2}$$

This notebook demonstrates analytic expansions, a small annealing search, and PyTorch gradient optimizations that reach high fidelity in the single-pair model.

---

### B. Linear 4-mode cluster (notebook: `4mode_cluster.ipynb`)

**Physical setup (initial):**

* 4 modes arranged in a linear chain (or square variant): nodes 0,1,2,3.
* Candidate edges (SPDCs) between nearest neighbors (and optionally diagonals).
* Local phases and beam-splitter mixing allowed.

**Target state (two complementary descriptions provided in the notebook):**

1. **CV cluster target (Gaussian picture)**: target covariance/adjacency matrix for a 4-mode linear cluster state; the objective is to match the covariance $V(\text{params})$ to $V_\text{target}$ (MSE or normalized overlap).

2. **Discrete postselected picture (single-pair):** target expressed in terms of perfect matchings in the 4-mode two-photon space. The three perfect matchings are:

   * $M_1$: (0,1)(2,3)
   * $M_2$: (0,2)(1,3)
   * $M_3$: (0,3)(1,2)
   
   Amplitudes $A_k$ are products of relevant edge weights and the fidelity can be computed on the $(A_1, A_2, A_3)$ vector.

**Notes:** The notebook contains both the Gaussian simulation plan and a discrete perfect-matching toy model; it shows how to switch between objectives and how QUBO chooses edges for each modeling choice.

---

### C. Star-shaped repeater node (notebook: `star_repeater.ipynb`)

**Physical setup (initial):**

* Four leaves (L0..L3) representing outgoing channels; pairwise SPDC edges among the four leaves (complete graph of 4 nodes → 6 edges).
* Central memory is treated conceptually but the detection/post-selection is on the 4 leaves.
* Continuous knobs per included edge and global/local mode phases; optional beam splitters.

**Three target states used (explicit):**

* **T1** — *Pair-Bell style:* concentrate amplitude on matching $M_1 = (L0,L1) \& (L2,L3)$: target vector $\mathbf{t}_1 = (1,0,0)$
* **T2** — *W-like superposition:* equal contributions of the three perfect matchings: $\mathbf{t}_2 = (1,1,1)/\sqrt{3}$
* **T3** — *Two-match superposition:* mixture of $M_1$ and $M_3$: $\mathbf{t}_3 = (1,0,1)/\sqrt{2}$

**How outputs are computed (single-pair):**

* Edge complex weights $w_e = \kappa_e\sqrt{P_e}e^{i\varphi_e}$
* Matching amplitudes: $A_1 = w_{01}w_{23}$, $A_2 = w_{02}w_{13}$, $A_3 = w_{03}w_{12}$
* Fidelity to normalized target $\mathbf{t}$: 

$$F = \frac{|\mathbf{t}^\dagger \mathbf{A}|^2}{\sum |A_k|^2}$$

This notebook demonstrates that different discrete skeletons are selected by QUBO for different targets and that continuous optimization tunes each skeleton to high fidelity.

---

## 3. QUBO formulation — from scratch

**Goal.** Encode discrete design choices $x_i \in \{0,1\}$ (include component $i$ or not) into a quadratic binary objective

$$\min_{x\in\{0,1\}^n} \; E(x) = x^T Q x$$

**Variable choices.**

* Each SPDC edge becomes a binary variable $s_e$
* Each candidate beam splitter becomes a binary variable $b_{ij}$ if you allow optional BS placement
* Optional extra variables encode routing/heralding choices

**Constructing Q (recipe).**

1. **Diagonal entries** $Q_{ii}$ — unary reward / cost for including a component:

   * If including the component is *desirable* (e.g., creates amplitude for the target), set $Q_{ii} = -\alpha$ (negative reward)
   * If expensive or undesirable, set $Q_{ii} = +\beta$ (positive cost)

2. **Off-diagonal** $Q_{ij}$ — pairwise interaction:

   * **Synergy (encourage pair)**: if $i$ & $j$ together enable desirable interference (e.g., two edges that form a perfect matching), set $Q_{ij} = -\gamma$
   * **Conflict (penalize pair)**: if $i$ & $j$ are incompatible (share the same physical port or violate hardware constraints), set $Q_{ij} = +\delta$

3. **Hard constraints** (e.g., at most $K$ active sources) encoded via quadratic penalty:

$$P \cdot \left(\sum_{i\in S} x_i - K\right)^2$$

   Expand and add to Q; choose $P$ large (5–50× typical Q magnitude) to enforce strongly.

4. **Higher-order preferences**: if you want k-body rewards, either approximate them with pairwise synergies, or reduce to quadratic by introducing auxiliary binary variables and equality constraints (standard reduction).

**Physics-guided calibration.**

* For small graphs: sample a few continuous optimizations with/without a component to estimate its utility. Set:

  * $Q_{ii} = -\lambda \cdot \text{gain}_i$ where $\text{gain}_i = \text{F}(i\text{ on}) - \text{F}(\text{none})$
  * $Q_{ij} = -\lambda \cdot \text{synergy}_{ij}$ where $\text{synergy}_{ij} = \text{F}(i,j\text{ on}) - \text{F}(i\text{ on}) - \text{F}(j\text{ on})$

**Scaling & solver choices.**

* Small graphs: brute force or classical simulated annealing (the notebooks include a Metropolis SA)
* Larger graphs: use `dimod` / D-Wave Ocean or hybrid solvers. The notebooks include export instructions for BQM format.

---

## 4. Continuous optimization — objective & gradients

**Parameterization (typical).**

* Per edge: raw variable $r_e$ → pump amplitude $P_e = \text{softplus}(r_e)$ (ensures positivity)
* Per edge: source phase $\varphi_e$ (real unconstrained)
* Per mode: local phase $\phi_i$ (real)
* Beam splitter angle: raw $t$ → $\theta = \text{sigmoid}(t) \cdot (\pi/2)$ to keep $\theta \in [0,\pi/2]$

**Complex coefficient per source**

$$c_{ij} = \kappa_{ij}\sqrt{P_{ij}} \, e^{i\varphi_{ij}}$$

**Beam splitter model (creation operators)**

$$a^\dagger \mapsto \cos\theta \, a^\dagger + \sin\theta \, b^\dagger, \quad b^\dagger \mapsto -\sin\theta \, a^\dagger + \cos\theta \, b^\dagger$$

**General objective (single-pair perfect-matching picture)**

Given final coefficients (matching amplitudes) $C_k(\vec{p})$, and a (normalized) complex target vector $\mathbf{t}$, the fidelity is:

$$F(\vec{p}) = \frac{\left|\mathbf{t}^\dagger \mathbf{C}(\vec{p})\right|^2}{\sum_k |C_k(\vec{p})|^2}$$

For the triangle case $\mathbf{t} = \frac{1}{\sqrt{3}}(1,1,1)$ and the formula reduces to the previously shown expression.

**Gradient formula (Wirtinger)**

Let $S = \sum_k C_k$, $N = \sum_k|C_k|^2$, and $F = |S|^2/(3N)$. Then for each complex coefficient:

$$\frac{\partial F}{\partial C_k^*} = \frac{1}{3} \, \frac{S \, N - |S|^2 C_k}{N^2}$$

Chain-rule through $C_k(\vec{p})$ gives gradients with respect to physical knobs. In practice we rely on **autodiff (PyTorch/JAX)** rather than hand-coding derivatives.

**Optimizers & hyperparameters**

* Use **Adam** (lr ~1e-3 to 1e-2) for noisy landscapes, then refine with **L-BFGS** for local minimization
* Initialization: small random phases, small equal pump amplitudes. For skeletons returned by QUBO, initialize excluded edges with near-zero pump (or clamp them to zero)
* Regularization: to avoid runaway pump growth (F invariant under uniform scale), either fix total pump budget or add a penalty on total pump power / prefer higher count rate via multi-objective:

$$\mathcal{L} = 1 - F - \lambda \cdot \frac{N}{N_\text{ref}}$$

where $N$ is the postselected norm (proxy for counts).

**Robustness & noise**

* Add Monte-Carlo phase/power perturbations to test stability
* Prefer solutions with high median fidelity under small random perturbations

---

## 5. Implementation details

**Language & libs (tested in Colab)**

* Python 3
* `numpy`, `matplotlib`, `networkx`, `torch` (PyTorch), `scipy` (optional), `strawberryfields` (for full-Fock validation; optional)
* Notebooks prepared for Google Colab (GPU accelerator recommended for faster PyTorch)

**Directory / notebook layout**

* `notebooks/triangle_photonic.ipynb` — triangle pipeline, analytic derivations, SA + PyTorch tuning, visualization. Produces `triangle_qubo_opt_summary.json` and `figures/*.png`
* `notebooks/4mode_cluster.ipynb` — 4-mode pipeline (Gaussian & discrete variants). Produces covariance comparisons and visualizations
* `notebooks/star_repeater.ipynb` — star repeater, three targets, discrete search and continuous optimization. Saves `figures/T1_optimized.png`, etc.
* `optimized_graphs/` — saved PNGs for each experiment & target
* `README.md` (this file)

**How to run (Colab)**

1. Upload notebooks to Colab (or open directly)
2. Optionally set Runtime ▶ Change runtime type ▶ Hardware accelerator: **GPU (T4)**
3. Install dependencies (if not present):

   ```bash
   !pip install torch torchvision networkx matplotlib numpy
   # Optional for full-fock:
   !pip install strawberryfields
   ```
4. Run cells from top to bottom. Visual outputs and PNGs are saved automatically

**Outputs saved automatically**

* Per-experiment summary JSON (best skeletons, optimized parameters, fidelity and norms)
* Figures (network visualizations, robustness plots) as PNGs in `figures/` or `optimized_graphs/`

---

## 6. Validation & caveats (what to check before lab implementation)

1. **Single-pair approximation** — The algebraic perfect-matching model assumes single-pair emission (weak pumping). In realistic SPDC, higher-order emissions matter. Always validate top designs with a **full Fock simulator** (Strawberry Fields or QuTiP), computing post-selected fidelities including vacuum and double-pair terms.

2. **Loss & detector inefficiency** — Include detector efficiencies and channel loss in the simulation; they change optimal pump settings (often favor lower pumps to reduce double-pair noise). See notebooks for Strawberry Fields validation snippets.

3. **Mapping abstract pumps to lab units** — `P_e` in the notebook is an abstract amplitude. Calibrate $P_e \leftrightarrow$ squeezing parameter $r \leftrightarrow$ pump power (mW) for your crystals from calibration data.

4. **Hardware constraints** — Some skeletons may require routing or interferometers not available on a given bench. Encode these as hard constraints in QUBO.

5. **QUBO weight sensitivity** — We hand-tune Q matrix magnitudes for small graphs. For larger graphs, use surrogate sampling or ML to estimate reliable Q entries.

6. **Statistical robustness** — For experimental deployment, run Monte-Carlo over parameter noise and detector variation; prefer designs with plateau-like high-fidelity regions.

---

## 7. Reproducibility, results, contact & license

**Reproducibility**

* Random seeds used in notebooks are printed and set at the top (`SEED`/`RNG_SEED`). To reproduce exact runs, set the same seeds and run the full notebook end-to-end.

**Example outputs**

* Sample optimized parameters (from runs): pump amplitudes, phases, and BS angles are saved per skeleton in summary JSON files (see `triangle_qubo_opt_summary.json`, or similar files saved by notebooks).

**Contact**

* For questions regarding code, experiments, or reproducibility, contact the author(s) at the address shown in the repository (or open an issue on the GitHub repo).

**License**

* Suggested: **MIT License**. Add `LICENSE` file if you plan to share publicly.

---

## 8. Suggested next steps (experimental & computational)

* Validate best designs with **Strawberry Fields** full-Fock simulation including loss & detector POVMs
* Optimize with a **multi-objective** combining fidelity and expected coincidence rate
* Scale to 5–6 nodes and explore surrogate methods to build QUBO automatically (pairwise synergy estimation from small-budget sampling)
* Port the discrete QUBO to a quantum annealer (D-Wave Leap) or hybrid sampler for larger graphs

---

**Acknowledgements / inspiration.** Ideas and graph-based formulation follow recent works leveraging graph/network representations for photonic experiment design (Xuemei Gu and collaborators). This repository demonstrates a practical hybrid pipeline and provides reproducible code for further research.
