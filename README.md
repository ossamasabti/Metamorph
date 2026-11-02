# Metamorph

Bayesian calibration and counterfactual design of a multi-compartment model for
nanoparticle transport, where the particle changes state as it circulates.

Status: work in progress (phases 1 and 2 of 6: forward model, synthetic data). The full README
(motivation, results, references) is written in phase 6.

## Installation

```
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -e .[dev]
pytest
python scripts/phase1_forward.py
python scripts/phase2_synthetic_data.py
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

## License

MIT, see `LICENSE`.
