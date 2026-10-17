# Indoor Li-Fi with Wi-Fi fallback: simulation

Course project, *Wireless Mobile Communication*, Review 2 (implementation, results, performance analysis,
comparison, demo).

The project simulates one room with a Li-Fi access point (a ceiling LED) and a Wi-Fi access point.
A user walks around the room, and people occasionally block the light. We compare three ways of
connecting: **Wi-Fi only**, **Li-Fi only**, and a **hybrid** that uses Li-Fi when it is good and falls
back to Wi-Fi when it is not, with hysteresis to avoid switching back and forth too often.

Everything is pure Python (numpy, scipy, matplotlib, pandas, pytest). No hardware is needed.
`lifi_sim.py` is the original prototype and is kept unchanged as the reference. The `lifi/`
package is the refactored and extended version.

> New to the topic? Read **[EXPLAINER.md](EXPLAINER.md)** first. For the numbers, see
> **[results/results_summary.md](results/results_summary.md)**.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q                    # 60 tests, ~1 min (includes the regression test)
.venv/bin/streamlit run app.py                   # interactive demo, see 'Running the demo'
.venv/bin/python -m lifi run --all               # all experiments + summary, ~15 s
.venv/bin/python -m lifi run --all --quick       # fewer seeds/bits/grid points, ~8 s
.venv/bin/python -m lifi run --exp E3            # one experiment (several allowed: --exp E4 E5)
.venv/bin/python -m lifi summary                 # rebuild results_summary.md from existing CSVs
```

Options: `--out DIR` (default `results`), `--seed N` (default 42). Run times were measured on
an ordinary laptop CPU.

## Folder structure

```
lifi_sim.py                 original prototype (unchanged, reference only)
lifi/
  config.py                 Config dataclass: every parameter with unit and source
  channel.py                Lambertian LOS gain, received power, SNR, link rate
  ber.py                    Q function, OOK / M-PAM BER, Monte Carlo
  wifi.py                   Wi-Fi path loss, shadowing, rate
  mobility.py               random waypoint, 2-state Markov blockage
  hybrid.py                 traces, link-selection policies, metrics
  experiments.py            E0-E8 and S1 (each writes a CSV and a PNG)
  plots.py                  shared figure style (300 dpi, fixed colours per policy)
  summary.py                builds results/results_summary.md from the CSVs
  __main__.py               command line interface
tests/test_lifi.py          unit tests and the regression test
tests/test_userdata.py      tests for user-data validation, Wi-Fi fitting, custom traces
tests/test_app.py           Streamlit AppTest smoke tests for the demo
app.py                      Streamlit demo app (entry point)
demo/                       demo helpers: content.py (reads results/docs), figs.py (Plotly figures),
                            make_examples.py (regenerates data/examples)
lifi/userdata.py            user-data helpers: CSV validation, Wi-Fi fit, traces from user path/blockage
data/examples/*.csv         SYNTHETIC sample files for the 'Your Data' tab (not real measurements)
results/data/*.csv          raw numbers for every experiment
results/figures/*.png       figures (300 dpi)
results/results_summary.md  markdown tables, ready to paste into slides or a report
EXPLAINER.md                plain-English explanation of every experiment
```

## Experiments

| ID | Question | Figure | Data |
|---|---|---|---|
| E0 | Prototype baseline: single run, seed 42 (regression reference) and a 60 s timeline | `E0_baseline_timeline.png` | `E0_baseline_seed42.csv` |
| E1 | Received power and SNR across the room, one LED | `E1_power_snr_heatmap.png` | `E1_power_snr_grid.csv` |
| E2 | BER vs distance (OOK, 4-PAM, 8-PAM); Monte Carlo vs theory for all three | `E2_ber_theory_vs_mc.png` | `E2_ber_vs_distance.csv`, `E2_ber_monte_carlo.csv` |
| E3 | Coverage (SNR ≥ 15 dB) and corner SNR vs LED height, semi-angle, FOV, bandwidth, power | `E3_parameter_sweeps.png` | `E3_parameter_sweeps.csv` |
| E4 | Throughput and outage vs blockage probability | `E4_blockage_sweep.png` | `E4_blockage_sweep.csv` |
| E5 | Hysteresis threshold sweep (up 10–25 dB, down 5–20 dB) | `E5_hysteresis_sweep.png` | `E5_hysteresis_sweep.csv` |
| E6 | **Headline numbers**: 30 independent seeds, mean ± 95% CI | `E6_statistics.png` | `E6_per_seed.csv`, `E6_statistics_summary.csv` |
| E7 | Policy comparison: Wi-Fi, Li-Fi, single-threshold ×2, hysteresis, ideal upper bound | `E7_policy_comparison.png` | `E7_per_seed.csv`, `E7_policy_comparison.csv` |
| E8 | (stretch) 2×2 LED grid, strongest-LED association, no interference | `E8_led_grid_coverage.png` | `E8_led_grid_coverage.csv` |
| S1 | Sensitivity: headline comparison with each optional fix switched on | (table only) | `S1_fix_flag_sensitivity.csv` |

**Statistics.** E4–E7 use 30 independent runs. Each run has its own random generator, spawned
from the master seed with `numpy.random.SeedSequence(42).spawn(30)`. Within an experiment, every
policy and parameter value is tested on the *same* 30 random realisations (common random numbers),
so differences between policies are not caused by luck of the draw. Confidence intervals are
Student-t 95% intervals over the 30 runs.

**Regression.** `tests/test_lifi.py::test_regression_matches_prototype_seed42` checks that the
default config with seed 42 reproduces the prototype exactly (to 12 significant digits):
Wi-Fi only 120.0 Mbps / 0.00% outage, Li-Fi only 438.3 Mbps / 10.73%, Hybrid 407.8 Mbps / 6.60%,
67 switches, 75.0% time on Li-Fi. In the prototype, the OOK Monte Carlo ran on the same global
generator *before* the hybrid simulation, so the hybrid result depends on those draws.
`experiments.prototype_baseline()` replays the same 13 × 2,000,000 draws first so the random
stream is identical. E0 reports this single run. The headline table uses E6 (30 independent seeds),
because one run is a single sample.

## Model summary

- **Li-Fi channel.** Line-of-sight Lambertian model. The LED points down and the photodiode
  points up, so the irradiance angle equals the incidence angle.
  H(0) = (m+1)·A/(2πd²) · cos^m(φ) · T_s · g(ψ) · cos(ψ), with g = n²/sin²(FOV) and H = 0 outside the FOV.
- **Li-Fi SNR.** Electrical SNR = (R·P_r)² / (2q(R·P_r + I_bg)B + 4kTB/R_L).
- **Rates (both links).** Rate = B · min(log2(1 + SNR/Γ), 6 bit/s/Hz), with gap Γ = 3 dB.
- **Wi-Fi.** PL = 40 + 30·log10(d) + X_σ dB (σ = 4 dB), 20 dBm transmit power, noise = −174 + 10·log10(B) + 7 dB.
- **BER.** Unipolar Gray-coded M-PAM: BER ≈ 2(M−1)/(M·log2M) · Q(√SNR/(M−1)). For OOK this is Q(√SNR).
- **Mobility.** Random waypoint at 1 m/s with no pause. Blockage is a 2-state Markov chain
  (enter with p_in = 0.01 per 0.1 s step, leave with p_out = 0.1, so blocked 9.1% of the time
  on average, mean block length 1 s).
- **Hybrid policy.** Start on Wi-Fi. Move to Li-Fi when Li-Fi SNR > 15 dB, return when it is
  < 10 dB. Each switch costs 0.3 s with zero throughput.
- **Metrics.** Average throughput (Mbps). Outage = % of time steps with rate < 1 Mbps.
  Coverage = % of room grid points with SNR ≥ 15 dB.

## Parameters

Sources: **[K04]** T. Komine and M. Nakagawa, "Fundamental analysis for visible-light communication
system using LED lights", *IEEE Trans. Consumer Electronics* 50(1), 2004. **[G19]** Z. Ghassemlooy,
W. Popoola and S. Rajbhandari, *Optical Wireless Communications*, 2nd ed., CRC Press, 2019.
**[R02]** T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd ed., 2002.
"Assumption" means the value was chosen for this project and not taken from a specific source.
Each value is also commented in `lifi/config.py`.

| Parameter (config field) | Value | Unit | Source |
|---|---:|---|---|
| Room side length, square room (`room`) | 6 | m | assumption |
| LED height (`h_led`) | 3 | m | assumption ([K04] room height 3 m) |
| Receiver height (`h_rx`) | 0.85 | m | [K04] |
| LED optical transmit power (`p_t`) | 1 | W | assumption |
| LED semi-angle at half power (`half_angle_deg`) | 60 | deg | typical value [G19]; assumption |
| Photodiode area (`a_pd`) | 1e-4 | m² | [K04] |
| PD responsivity (`resp`) | 0.54 | A/W | typical Si PIN ([K04] uses 0.53) |
| Receiver field of view (`fov_deg`) | 60 | deg | assumption |
| Concentrator refractive index (`n_ref`) | 1.5 | - | [K04] |
| Optical filter gain (`t_s`) | 1 | - | assumption (ideal filter) |
| Ambient-light photocurrent (`i_bg`) | 5.1e-3 | A | [K04] |
| Load resistance (`r_load`) | 50 | Ω | assumption |
| Temperature (`temp`) | 300 | K | assumption |
| Li-Fi bandwidth (`b_lifi`) | 100 | MHz | assumption |
| Wi-Fi bandwidth (`b_wifi`) | 20 | MHz | IEEE 802.11 20 MHz channel |
| Spectral-efficiency cap (`max_se`) | 6 | bit/s/Hz | assumption (~64-QAM) |
| SNR gap to Shannon (`gap_db`) | 3 | dB | assumption |
| Wi-Fi AP position (`wifi_xy`) | (0, 0) | m | assumption (room corner) |
| Wi-Fi transmit power (`wifi_ptx_dbm`) | 20 | dBm | assumption (typical) |
| Wi-Fi path loss at 1 m (`wifi_pl0_db`) | 40 | dB | free-space loss at 2.4 GHz |
| Wi-Fi path-loss exponent (`wifi_pl_exp`) | 3 | - | assumption (indoor range 2–4 [R02]) |
| Wi-Fi noise figure (`wifi_nf_db`) | 7 | dB | assumption |
| Wi-Fi shadowing std (`wifi_shadow_db`) | 4 | dB | assumption (indoor 3–6 dB) |
| Simulated time per run (`sim_time`) | 300 | s | assumption |
| Time step (`dt`) | 0.1 | s | assumption |
| User speed, random waypoint (`speed`) | 1 | m/s | assumption |
| P(enter blockage) per step (`p_block_in`) | 0.01 | - | assumption |
| P(leave blockage) per step (`p_block_out`) | 0.1 | - | assumption |
| Hysteresis up threshold (`thr_hi_db`) | 15 | dB | assumption |
| Hysteresis down threshold (`thr_lo_db`) | 10 | dB | assumption |
| Handover outage per switch (`switch_cost_steps`) | 3 (= 0.3 s) | steps | assumption |
| Outage rate threshold (`outage_mbps`) | 1 | Mbps | assumption |
| Coverage SNR threshold (`coverage_snr_db`) | 15 | dB | assumption |
| Master random seed (`seed`) | 42 | - | prototype |
| Independent runs for statistics (`n_seeds`) | 30 | - | assumption |

## Assumptions made while building this project

- Monte Carlo BER in E2 uses its own generator (seed 42) and 3×10⁶ bits per point (3×10⁵ in `--quick`).
  Points with zero bit errors are not plotted; the CSV reports the error count of every point.
- E4 varies only p_in (from 0 to 0.1) and keeps p_out = 0.1, so blocked time ranges from 0 to 50%.
- E5 uses a 1 dB grid with down < up strictly. E7's single-threshold policies use 10 dB and
  15 dB, the two edges of the hysteresis band.
- The "Ideal" policy in E7 (best link at every step, no switching cost) is an upper bound for
  reference, not a realistic policy.
- E3 coverage uses a 121 × 121 grid that includes the room edges (61 × 61 in quick mode). The
  corner is (0, 0); when it falls outside the FOV, its SNR is reported as "no LOS" (NaN in the CSV).
- E8 places the four LEDs at (1.5, 1.5), (4.5, 1.5), (1.5, 4.5) and (4.5, 4.5) m. It compares
  1 W per LED (4 W total) and 0.25 W per LED (equal 1 W total) against the single 1 W LED.
- `--quick` lowers the number of seeds in E4 (10) and E5 (5), the Monte Carlo bits and the grid size.
  E6 and E7 always use 30 seeds because they are cheap. Quick-mode numbers are for checking only.

## Modelling assumptions and limitations

- **Line of sight only.** Wall reflections (NLOS) are ignored. They would add some power near
  the walls and spread the pulse, which limits the usable bandwidth.
- **No inter-cell interference.** In E8 the user hears only the strongest LED, and the other LEDs
  add no interference, which is optimistic. With interference, cell edges would get much worse.
- **Simplified Wi-Fi.** One AP and one user. No walls, no other users or contention, no MAC
  overhead and no co-channel interference. With these parameters the SNR is always high enough to
  reach the rate cap, so Wi-Fi is a constant 120 Mbps (see Known issues).
- **Capped spectral efficiency** of 6 bit/s/Hz on both links, with a fixed **3 dB SNR gap** to
  Shannon. No real modulation-and-coding table is used.
- The receiver always points straight up: no device tilt or rotation. The user's body only
  appears through the blockage process.
- The LED's own frequency response (its modulation bandwidth limit) is not modelled. B = 100 MHz is assumed usable.
- The hybrid controller knows the Li-Fi SNR perfectly and instantly: no measurement noise or
  delay, and no signalling overhead apart from the fixed 0.3 s handover gap.
- Random-waypoint users spend more time near the room centre than near the walls. Because the LED
  is at the centre, this favours Li-Fi.
- Single user, 2-D movement at a fixed receiver height of 0.85 m.

## Known issues / possible improvements

These come from reviewing the prototype. Where an issue changes results, the original behaviour
is kept by default (so the regression test still holds). Any fix sits behind a config flag that is
**off by default**. S1 shows what each flag does to the headline numbers (see `results_summary.md` §8).

1. **Wi-Fi is always saturated, so shadowing has no effect.** Even at the far corner (8.5 m)
   with +16 dB (4σ) shadowing, the Wi-Fi SNR is about 30 dB, above the ~21 dB needed to hit the
   6 bit/s/Hz cap. So "Wi-Fi only" is exactly 120 Mbps with 0% outage in every run, and the random
   shadowing never changes anything. The Wi-Fi model makes the fallback look perfect.
   *Flags:* `wifi_mac_overhead=True` scales Wi-Fi by `wifi_mac_efficiency` (0.6, assumption) to
   account for MAC/protocol overhead; Wi-Fi only then gives 72 Mbps. A realistic fix (walls,
   contention, interference) is not implemented.
2. **Shadowing is independent at every 0.1 s step.** Real shadowing is correlated over metres of
   movement. *Flag:* `wifi_correlated_shadowing=True` uses an AR(1) (Gudmundson) model with a
   2 m decorrelation distance. Because of issue 1, it changes nothing in the results.
3. **The Shannon formula is optimistic for Li-Fi.** Li-Fi uses intensity modulation with direct
   detection (IM/DD), so the signal must be real and non-negative. Its capacity is below
   B·log2(1+SNR). *Flag:* `lifi_imdd_capacity=True` uses the bound B·log2(1 + e/(2π)·SNR/Γ),
   equivalent to 3.6 dB less SNR. Li-Fi-only throughput then drops from 434.6 to 365.4 Mbps.
4. **The noise model is simplified.** Thermal noise is 4kTB/R_L only, with no amplifier noise
   figure, FET channel noise, or the 0.562 noise-bandwidth factor used in [K04]. So the SNR is
   somewhat optimistic. Ambient-light shot noise dominates here, which is also why SNR scales
   with the square of optical power (relevant in E8). Not changed.
5. **All hybrid outage comes from handovers.** Outage counts the 0.3 s zero-rate gap after every
   switch. Wi-Fi never fails, so every hybrid outage step is a handover gap. This is a modelling
   choice, but it explains why the hybrid has *more* outage than Li-Fi only when blockage is rare (E4).
6. **The 15/10 dB thresholds sit far above the rate crossover.** Li-Fi beats 120 Mbps Wi-Fi from
   about 4 dB SNR, so the hybrid sometimes uses Wi-Fi while Li-Fi would give 150–300 Mbps (visible
   in the E0 timeline). E5 shows lower thresholds do better (10/5 dB: 426.5 vs 397.9 Mbps).
   The thresholds were **not** retuned, so the reported numbers stay those of the stated design.
7. **The prototype's Monte Carlo plot hid zero-error points.** With 2×10⁶ bits at 15–16 dB the
   expected number of errors is ≪ 1, so those points measured BER = 0, which the plot drew at
   1e-7. E2 plots only points with errors and stores the error count of every point.
8. **The prototype's hybrid result depended on the BER Monte Carlo.** This was a side effect of
   the shared global generator; the refactor passes a generator explicitly. It is reproduced in
   E0 only for the regression test.
9. **Small start-up biases.** Every run starts unblocked, rather than drawing the first state
   from the chain's long-run distribution, and the hybrid starts on Wi-Fi, so it pays one handover
   in the first 0.3 s if Li-Fi is good. Both effects are below 0.2% of a 300 s run.
10. **The BER formula is the standard nearest-neighbour approximation** (BER ≈ SER/log2 M).
    Monte Carlo agrees within 6% at every point with at least 100 errors; the gap is largest at low SNR.

## Tests

`python -m pytest -q` runs 60 tests. `tests/test_lifi.py` (18, unchanged) covers:

- Channel gain equals a hand calculation of the textbook formula at (4, 4) m.
- Gain is exactly zero just outside the FOV and at the corner, and decreases with distance.
- BER decreases monotonically with SNR for M = 2, 4, 8; BER(OOK) = Q(√SNR); higher-order PAM
  is worse at equal SNR.
- Monte Carlo BER is within 15% of theory for six (scheme, SNR) points, each with ≥ 200 errors.
- The hybrid never switches while the SNR stays between the two thresholds. Handover penalty
  timing is checked. A single threshold ping-pongs under noise where hysteresis does not.
- The regression test against the prototype (above).

`tests/test_userdata.py` (32) checks that the Wi-Fi fit recovers known parameters from synthetic
noisy data (within 4 standard errors for the geometry) and exactly from noise-free data; that the
validators reject bad files (missing columns, non-numeric or empty cells, fewer than 6 Wi-Fi points,
positions outside the room, centimetres, positive or impossible RSSI, non-increasing time, `blocked`
not 0/1, binary garbage); and that with no uploads the traces and the default `Config` are unchanged.
`tests/test_app.py` (10) loads the app with `streamlit.testing.v1.AppTest` and checks that the
three pages render without exceptions, the headline metrics and expanders are present, the
threshold check and channel reset work, the seed-42 replay shows the prototype's numbers, every
experiment in the gallery loads, the app opens with simulated data only, and the sample data
work end to end and switch off again.

## Running the demo

```bash
.venv/bin/pip install -r requirements.txt        # adds streamlit and plotly
.venv/bin/streamlit run app.py                   # opens http://localhost:8501
```

The demo works offline (Streamlit bundles Plotly's JavaScript). `.streamlit/config.toml` fixes one
light theme: white background, near-black text and a single deep-blue accent. It is used even if
the viewer's computer is in dark mode, because light pages read better on projectors. All custom
HTML (pipeline diagram, red banner) sets its own text and background colours, and every visible
text item was checked with an automated WCAG contrast scan (all at least AA, 4.5:1). Plotly charts
use white backgrounds with near-black axis text. The matplotlib figures in `results/figures` were
already white with dark text and are unchanged. If a results file is missing, the app says so and
tells you to run `python -m lifi run --all`.

The app has three pages:

| Page | What it shows |
|---|---|
| Project | Title with a one-line note that this is a simulation, four plain sentences, the pipeline diagram (mobility + blockage → Li-Fi channel + Wi-Fi → SNR → link selector → metrics) and the headline numbers from `results/data/E6_statistics_summary.csv`: speed and time disconnected for Wi-Fi only, Li-Fi only and Hybrid (95% intervals in a caption), plus one sentence stating the finding, worded from the numbers. Collapsed below: **More experiments** (E0–E8, S1: figure, explanation from EXPLAINER.md, CSV downloads), **Limitations and honest notes** (README sections), **Policy comparison (E7)**. |
| Live demo | Seed and "light blocked" controls (the slider shows the long-run share of time blocked). The room view shows the light, the Wi-Fi router and the user with a 10 s trail; the marker's colour and shape show Li-Fi / Wi-Fi / light blocked / switching. Also: Play/Pause, a time slider, a small table of speed and time disconnected for this run, and speed over time with blocked periods shaded. The collapsed **Advanced settings and channel explorer** holds the hysteresis levels, switching penalty, the seed-42 replay (reproduces 120.0 / 438.3 / 407.8 Mbps, 67 switches), and the LED/receiver sliders with the probe table (coverage, SNR, BER per modulation), the power and SNR heatmaps and BER vs distance. Each technical term has a one-line explanation there. |
| Your data (optional) | Three steps: 1 Wi-Fi measurements, 2 walking path, 3 blockage timeline. Then results: the same views as the live demo plus a table comparing with the default simulation. "Load sample files", empty templates and sample-file downloads are kept. |

**Speed.** Room maps and runs are cached (`st.cache_data`), and one uncached run takes under 10 ms.
The 30-seed statistics are always read from `results/data`, never recomputed live.

### Using the "Your data" page

Everything here is **off by default**, and the app always opens with the simulated version.
Whenever any user data is in use, a red **"Using user-provided data"** banner appears on each page
that uses it (Live demo and Your data).

1. **Try it without files:** click **"Load sample files"**, then tick the "Use ..." boxes. The samples
   in `data/examples/` are **synthetic**: they are generated by `python -m demo.make_examples` from a
   known model (signal at 1 m = −40 dBm, n = 2.8, σ = 3 dB), and are not real measurements. An
   uploaded file replaces the sample of the same kind.
2. **Wi-Fi measurements**: CSV `x_m,y_m,rssi_dbm`, at least 6 points inside the 6 × 6 m room, with
   RSSI in dBm (−120 to −10), plus the router position. The app fits
   RSSI(d) = RSSI(1 m) − 10·n·log10(d) by least squares. It shows a table of the fit (signal at 1 m,
   fade rate n, random spread = residual standard deviation, R²) beside the simulation defaults,
   and a scatter with the fitted line. Router power and the 1 m path loss cannot be separated, so
   their combination is fitted. **"Use this Wi-Fi fit in the simulation"** passes the fit to `wifi.py`
   through a copy of the config (`wifi_ptx_dbm` = RSSI(1 m), `wifi_pl0_db` = 0, `wifi_pl_exp` = n,
   `wifi_shadow_db` = σ, `wifi_xy` = router). The defaults in `config.py` are never edited.
3. **Walking path**: CSV `t_s,x_m,y_m` with increasing time (s) and positions (m) inside the room,
   at most 1800 s. It is linearly resampled to the 0.1 s step.
4. **Blockage timeline**: CSV `t_s,blocked` with `blocked` = 0 or 1. Each value holds until the next
   row, resampled to 0.1 s.
5. **Results** runs Wi-Fi only, Li-Fi only and Hybrid with whichever inputs are ticked, and compares
   them in a table with the default simulation for the same seed.

Invalid files never crash the app; the error says what is wrong and on which line. Uploaded files
are kept only in the browser session's memory and are never written to disk.

**Choices made for the demo.**
- **Page structure.** The three pages are tabs in one script, so the red banner and user data stay
  consistent across pages in a single run.
- **Headline display.** The Project page shows the six headline numbers as large metrics arranged
  as a three-column table (one column per scheme), instead of a separate grid table, so they are not
  shown twice. The ± values are in a caption, and E7 shows means only (its intervals are in the CSV).
- **Live demo blockage control.** It offers the E4 blockage values, labelled by the long-run share of
  time blocked (0–50%). The page updates immediately, so there is no Run button.
- **Advanced settings.** The hysteresis levels, switching penalty and replay checkbox sit in the
  advanced expander below the results. Their values are kept in session state, so changing them
  reruns the demo above.
- **Room view.** It shows only the last 10 s of the path.
- **Your data run settings.** That page uses the default hybrid settings (15/10 dB, 0.3 s, p_in = 0.01)
  and its own seed. The Live demo's advanced settings do not affect it.
- **Scope of the Wi-Fi fit.** It applies to the Live demo and Your data pages only; the precomputed
  experiments never change.
- **Run length.** The user path's duration if used, otherwise the blockage timeline's duration if used,
  otherwise 300 s. Path and blockage are aligned at their first timestamps, and a shorter timeline
  holds its last value.
- **Fit details.** Distances under 1 m count as 1 m in the fit, as in the simulator. A walking speed
  above 3 m/s gives a warning but does not block the file.
- **Visual style.** No emojis or icons. Coloured callouts are used only for real warnings: the
  user-data banner, file errors and the threshold-order error.

## Known limits

- **Li-Fi remains simulated.** A handful of phone measurements gives only a rough Wi-Fi fit and is
  not a calibrated survey. Phone RSSI readings are coarse, device-dependent and averaged by the OS.
- With the simulator's noise floor (−94 dBm) and rate cap (120 Mbps), any RSSI above about
  −73 dBm saturates Wi-Fi. A typical indoor survey therefore still gives 120 Mbps Wi-Fi, which is
  the same limitation as Known issue 1.
- The demo's live runs are single random samples. Quote the 30-seed E6/E7 tables for results.

