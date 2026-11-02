# Metamorph

Bayesian calibration and counterfactual design of a multi-compartment model for
nanoparticle transport, where the particle changes state as it circulates.

Status: work in progress (phases 1 to 3 of 6: forward model, synthetic data,
Bayesian calibration). The full README (motivation, results, references) is
written in phase 6.

## Installation

```
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -e .[dev]
pytest
python scripts/phase1_forward.py
python scripts/phase2_synthetic_data.py
python scripts/phase3_calibration.py     # about 10 min on a laptop CPU
python scripts/phase3_mass_matrix.py     # optional, about 7 min
```

## Model

Blood (B) and five tissues (lung, liver, spleen, kidney, brain). Each particle is
either bare (b) or coated (c) with plasma proteins. Amounts are in percent of
injected dose (%ID), time in hours, rates in 1/h.

```
dB_b/dt       = -k_coat B_b - sum_i k_in[i,b] B_b + sum_i k_out[i] T_i,b - CL_renal B_b
dB_c/dt       =  k_coat B_b - sum_i k_in[i,c] B_c + sum_i k_out[i] T_i,c
dT_i,s/dt     =  k_in[i,s] B_s - k_out[i] T_i,s - [i = liver] CL_liver T_i,s
dA_renal/dt   =  CL_renal B_b
dA_hepatic/dt =  CL_liver (T_liver,b + T_liver,c)
```

## Model assumptions and limitations

1. **Effective coating rate.** `k_coat` is an effective rate for a functional
   change of state, namely the maturation of the hard protein corona that alters
   macrophage recognition and barrier crossing. It is deliberately slower than
   the initial protein adsorption, which happens within minutes, so that both
   states are visible on the time scale of the measurements.
2. **Irreversible coating, in blood only.** Particles do not lose their corona,
   and particles taken up by a tissue keep the state they had on entry.
3. **Linear kinetics.** All fluxes are first order in the amount: no saturation
   of macrophage uptake, no dose dependence.
4. **Amounts, not concentrations.** Organ volumes and blood flows are not
   represented; each rate constant lumps perfusion, permeability and binding.
   This is not a physiologically based (PBPK) model.
5. **Grouped parameters for identifiability.** Coated uptake is
   `k_in[i,c] = m_i k_in[i,b]`, with one multiplier `m_RES` shared by liver and
   spleen, a free `m_brain`, and `m_lung = m_kidney = 1`. Return rates are
   grouped as `k_out_fast` (lung, kidney) and `k_out_slow` (liver, spleen, brain).
6. **Clearance routes.** Renal filtration acts on bare particles in blood only;
   hepatic clearance acts on both states in the liver. No other elimination
   route is modelled.
7. **Closed mass balance.** Cleared amounts are tracked in two cumulative states,
   so the total amount equals the dose at all times.

## Synthetic experiment (phase 2)

Observed quantities (totals, bare and coated not separable), 3 independent
replicates each, multiplicative log-normal noise with sigma = 0.2:

| Observable | Sampling times (h) |
|---|---|
| Blood (B_b + B_c) | 5/60, 15/60, 30/60, 1, 4, 8, 24 |
| Urine (cumulative A_renal) | 5/60, 15/60, 30/60, 1, 4, 8, 24 |
| Lung, liver, spleen, kidney, brain | 1, 4, 24 |

A_hepatic is not observed. Data and true parameters: `data/`.

Additional assumptions of the synthetic experiment:

8. **Independent replicates.** Measurement errors are independent across
   replicates and times; inter-individual variability and within-animal
   correlation of serial blood samples are not modelled.
9. **One noise scale.** The same log-scale noise applies to all observables.

## Bayesian calibration (phase 3)

Log-normal priors from order-of-magnitude reasoning (medians deliberately not
equal to the true values), sampled in standardized log space; NUTS with a dense
mass matrix, 4 chains of 1000 warmup and 1000 draws. Diagnostics: 0 divergent
transitions, max R-hat 1.000, min bulk ESS 2450 (`results/phase3_summary.txt`).
93.1 % of the data points lie inside the 90 % posterior predictive band.

| Parameter | True | Prior median | Posterior mean | Posterior median | 90 % CI | True in CI | R-hat | ESS bulk |
|---|---|---|---|---|---|---|---|---|
| k_coat | 1.5 | 1 | 1.92 | 1.9 | [1.39, 2.51] | yes | 1.000 | 3386 |
| CL_renal | 0.8 | 0.5 | 0.921 | 0.914 | [0.785, 1.07] | yes | 1.000 | 4261 |
| CL_liver | 0.05 | 0.1 | 0.0529 | 0.0528 | [0.0438, 0.0625] | yes | 1.000 | 4104 |
| k_in_lung | 0.3 | 0.2 | 0.256 | 0.254 | [0.207, 0.313] | yes | 1.000 | 4817 |
| k_in_liver | 0.4 | 0.2 | 0.338 | 0.326 | [0.177, 0.54] | yes | 1.000 | 3714 |
| k_in_spleen | 0.1 | 0.2 | 0.087 | 0.084 | [0.045, 0.139] | yes | 1.000 | 3501 |
| k_in_kidney | 0.2 | 0.2 | 0.166 | 0.165 | [0.134, 0.205] | yes | 1.000 | 5111 |
| k_in_brain | 0.01 | 0.01 | 0.0118 | 0.0118 | [0.0085, 0.0153] | yes | 1.000 | 2964 |
| m_RES | 3 | 2 | 4.1 | 3.72 | [1.99, 7.5] | yes | 1.000 | 3543 |
| m_brain | 0.3 | 0.5 | 0.373 | 0.325 | [0.131, 0.775] | yes | 1.000 | 2450 |
| k_out_fast | 0.5 | 0.3 | 0.434 | 0.431 | [0.35, 0.535] | yes | 1.000 | 5269 |
| k_out_slow | 0.03 | 0.05 | 0.0349 | 0.0346 | [0.0273, 0.0437] | yes | 1.000 | 3614 |
| sigma | 0.2 | HalfN(0.5) | 0.196 | 0.195 | [0.171, 0.224] | yes | 1.000 | 3810 |

Well determined: clearance rates, k_coat, uptake into lung, kidney and brain,
both return rates. Poorly determined: m_brain, m_RES, k_in_liver, k_in_spleen,
because only bare + coated totals are observed.

## License

MIT, see `LICENSE`.
