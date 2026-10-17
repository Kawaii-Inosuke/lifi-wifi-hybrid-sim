"""
Li-Fi indoor simulation with Wi-Fi fallback (Wireless Mobile Communication project)

Modules
  1. Lambertian LOS channel model (LED -> photodiode)
  2. Received power, noise (shot + ambient + thermal) and SNR maps
  3. BER of OOK and M-PAM vs distance (theory) + Monte Carlo validation
  4. Hybrid Li-Fi / Wi-Fi link selection with a moving, blockage-prone user

Run:  python lifi_sim.py        (saves 4 PNG figures next to the script)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- parameters
ROOM = 6.0                 # room is ROOM x ROOM metres
H_LED, H_RX = 3.0, 0.85    # ceiling LED height, receiver (desk/phone) height
LED_POS = np.array([ROOM / 2, ROOM / 2])      # LED at ceiling centre

P_T = 1.0                  # transmitted optical power (W)
HALF_ANGLE = np.deg2rad(60)   # LED semi-angle at half power
A_PD = 1e-4                # photodiode area (1 cm^2)
RESP = 0.54                # responsivity (A/W)
FOV = np.deg2rad(60)       # receiver field of view
N_REF = 1.5                # lens refractive index
T_S = 1.0                  # optical filter gain
I_BG = 5.1e-3              # ambient-light photocurrent (A)
R_LOAD = 50.0              # load resistance (ohm)
TEMP = 300.0               # K
Q_E, K_B = 1.602e-19, 1.381e-23

B_LIFI = 100e6             # Li-Fi modulation bandwidth (Hz)
B_WIFI = 20e6              # Wi-Fi channel bandwidth (Hz)
MAX_SE = 6.0               # spectral-efficiency cap (bit/s/Hz) for both links
GAP_DB = 3.0               # SNR gap to Shannon (practical coding/modulation)

WIFI_POS = np.array([0.0, 0.0])               # Wi-Fi AP in a room corner


# ------------------------------------------------------------ 1. channel model
def lifi_gain(xy):
    """DC channel gain H(0) for receiver positions xy (N,2), vertical PD."""
    xy = np.atleast_2d(xy)
    dh = np.linalg.norm(xy - LED_POS, axis=1)
    dv = H_LED - H_RX
    d = np.hypot(dh, dv)
    cos_phi = dv / d                      # irradiance angle == incidence angle
    psi = np.arccos(cos_phi)
    m = -np.log(2) / np.log(np.cos(HALF_ANGLE))          # Lambertian order
    g_conc = N_REF ** 2 / np.sin(FOV) ** 2               # concentrator gain
    h = (m + 1) * A_PD / (2 * np.pi * d ** 2) * cos_phi ** m * T_S * g_conc * cos_phi
    return np.where(psi <= FOV, h, 0.0), d


def lifi_snr(xy, bw=B_LIFI):
    """Electrical SNR (linear) at positions xy."""
    h, _ = lifi_gain(xy)
    p_rx = P_T * h
    i_sig = RESP * p_rx
    shot = 2 * Q_E * (i_sig + I_BG) * bw
    thermal = 4 * K_B * TEMP * bw / R_LOAD
    return i_sig ** 2 / (shot + thermal), p_rx


# --------------------------------------------------------------------- 2. BER
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ber_pam(snr, M):
    """Gray-coded unipolar M-PAM (M=2 -> OOK) under an average-power constraint."""
    k = np.log2(M)
    return 2 * (M - 1) / (M * k) * qfunc(np.sqrt(snr) / (M - 1))


# ----------------------------------------------------------------- 4. Wi-Fi
def wifi_snr(xy, shadow_db=0.0):
    d = np.maximum(np.linalg.norm(np.atleast_2d(xy) - WIFI_POS, axis=1), 1.0)
    pl = 40 + 30 * np.log10(d) + shadow_db                 # log-distance, n=3
    p_rx_dbm = 20 - pl                                      # 20 dBm AP
    noise_dbm = -174 + 10 * np.log10(B_WIFI) + 7            # 7 dB noise figure
    return 10 ** ((p_rx_dbm - noise_dbm) / 10)


def rate(snr, bw):
    se = np.minimum(np.log2(1 + snr / 10 ** (GAP_DB / 10)), MAX_SE)
    return se * bw / 1e6                                    # Mbps


def random_waypoint(steps, dt, speed=1.0):
    pos = rng.uniform(0, ROOM, 2)
    wp = rng.uniform(0, ROOM, 2)
    out = np.empty((steps, 2))
    for i in range(steps):
        v = wp - pos
        dist = np.linalg.norm(v)
        if dist < speed * dt:
            pos, wp = wp, rng.uniform(0, ROOM, 2)
        else:
            pos = pos + v / dist * speed * dt
        out[i] = pos
    return out


def simulate_hybrid(T=300, dt=0.1, p_block_in=0.01, p_block_out=0.10,
                    thr_hi_db=15, thr_lo_db=10, switch_cost_steps=3):
    steps = int(T / dt)
    path = random_waypoint(steps, dt)
    blocked = np.zeros(steps, bool)
    for i in range(1, steps):                               # 2-state Markov blockage
        blocked[i] = (rng.random() < p_block_in) if not blocked[i - 1] \
            else (rng.random() > p_block_out)
    shadow = rng.normal(0, 4, steps)                        # Wi-Fi shadowing

    snr_l, _ = lifi_snr(path)
    snr_l = np.where(blocked, 0.0, snr_l)
    r_l = rate(snr_l, B_LIFI)
    r_w = rate(wifi_snr(path, shadow), B_WIFI)
    snr_l_db = 10 * np.log10(np.maximum(snr_l, 1e-12))

    # hybrid policy with hysteresis + switching penalty
    use_lifi = np.zeros(steps, bool)
    state, penalty, switches = False, 0, 0
    r_h = np.zeros(steps)
    for i in range(steps):
        new = state
        if not state and snr_l_db[i] > thr_hi_db:
            new = True
        elif state and snr_l_db[i] < thr_lo_db:
            new = False
        if new != state:
            switches += 1
            penalty = switch_cost_steps
        state = new
        use_lifi[i] = state
        r_h[i] = 0.0 if penalty > 0 else (r_l[i] if state else r_w[i])
        penalty = max(penalty - 1, 0)

    res = {}
    for name, r in (("Wi-Fi only", r_w), ("Li-Fi only", r_l), ("Hybrid", r_h)):
        res[name] = dict(avg=r.mean(), outage=100 * (r < 1.0).mean())
    res["Hybrid"]["switches"] = switches
    res["Hybrid"]["lifi_share"] = 100 * use_lifi.mean()
    return res, (r_w, r_l, r_h, blocked, dt)


# ===================================================================== main
if __name__ == "__main__":
    # ---- Fig 1: received power & SNR heatmaps
    g = np.linspace(0, ROOM, 121)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    snr_map, p_map = lifi_snr(pts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = ax[0].imshow(10 * np.log10(np.maximum(p_map, 1e-12) / 1e-3).reshape(X.shape),
                       origin="lower", extent=[0, ROOM, 0, ROOM], cmap="viridis",
                       vmin=-30, vmax=-10)
    ax[0].set_title("Received optical power (dBm)")
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(10 * np.log10(np.maximum(snr_map, 1e-12)).reshape(X.shape),
                       origin="lower", extent=[0, ROOM, 0, ROOM], cmap="plasma",
                       vmin=0, vmax=45)
    ax[1].set_title("SNR (dB), B = 100 MHz")
    plt.colorbar(im1, ax=ax[1])
    for a in ax:
        a.plot(*LED_POS, "w*", ms=14)
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig("fig1_power_snr_heatmap.png", dpi=150)
    plt.close()

    # ---- Fig 2: BER vs horizontal distance for OOK / 4-PAM / 8-PAM
    r_h = np.linspace(0, 4.2, 200)
    line = np.column_stack([LED_POS[0] + r_h, np.full_like(r_h, LED_POS[1])])
    snr_line, _ = lifi_snr(line)
    plt.figure(figsize=(6.5, 4.5))
    for M, lab in ((2, "OOK"), (4, "4-PAM"), (8, "8-PAM")):
        plt.semilogy(r_h, np.maximum(ber_pam(snr_line, M), 1e-15), label=lab)
    plt.axhline(1e-3, color="gray", ls="--", lw=0.8)
    plt.axvline(np.tan(FOV) * (H_LED - H_RX), color="red", ls=":", label="FOV edge")
    plt.xlabel("Horizontal distance from LED axis (m)")
    plt.ylabel("BER")
    plt.ylim(1e-12, 1)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title("BER vs distance (B = 100 MHz)")
    plt.tight_layout()
    plt.savefig("fig2_ber_vs_distance.png", dpi=150)
    plt.close()

    # ---- Fig 3: Monte Carlo validation of OOK theory
    snr_db = np.arange(4, 17, 1.0)
    n_bits = 2_000_000
    ber_mc = []
    for s in snr_db:
        snr = 10 ** (s / 10)
        bits = rng.integers(0, 2, n_bits)
        sigma = 1.0
        a = np.sqrt(snr) * sigma                 # mean signal amplitude (R*Pr)
        rx = 2 * a * bits + rng.normal(0, sigma, n_bits)
        ber_mc.append(np.mean((rx > a) != bits))
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy(snr_db, ber_pam(10 ** (snr_db / 10), 2), "b-", label="Theory  Q(sqrt(SNR))")
    plt.semilogy(snr_db, np.maximum(ber_mc, 1e-7), "ro", label="Monte Carlo")
    plt.xlabel("Electrical SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title("OOK: simulation vs theory")
    plt.tight_layout()
    plt.savefig("fig3_mc_validation.png", dpi=150)
    plt.close()

    # ---- Fig 4 + table: hybrid Li-Fi / Wi-Fi
    res, (r_w, r_l, r_hyb, blocked, dt) = simulate_hybrid()
    print(f"{'Scheme':<12}{'Avg Mbps':>10}{'Outage %':>10}")
    for k, v in res.items():
        print(f"{k:<12}{v['avg']:>10.1f}{v['outage']:>10.2f}")
    print(f"Hybrid switches: {res['Hybrid']['switches']}, "
          f"time on Li-Fi: {res['Hybrid']['lifi_share']:.1f}%")

    n = int(60 / dt)
    t = np.arange(n) * dt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]},
                           sharex=True)
    ax[0].plot(t, r_w[:n], label="Wi-Fi only", alpha=0.7)
    ax[0].plot(t, r_l[:n], label="Li-Fi only", alpha=0.7)
    ax[0].plot(t, r_hyb[:n], "k", lw=1.5, label="Hybrid")
    ax[0].set_ylabel("Throughput (Mbps)")
    ax[0].legend(ncol=3)
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(t, blocked[:n], step="pre", color="tab:red", alpha=0.6)
    ax[1].set_yticks([])
    ax[1].set_ylabel("Li-Fi\nblocked")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("fig4_hybrid_timeline.png", dpi=150)
    plt.close()
    print("Saved fig1-fig4.")
