# What this project does, in simple words

**The idea.** Li-Fi sends internet data using light. A ceiling LED flickers far too fast for your
eyes to notice, and a small light sensor (a photodiode) on your phone or laptop picks it up. Li-Fi
can be very fast, but it needs a clear line of sight: if a person walks in front of the light, or
you move too far away, the signal is lost. Wi-Fi is slower, but it goes around people and through
the room. So we simulate a **hybrid**: use Li-Fi when it is good, and switch to Wi-Fi when it is
not. Switching is not free (the link is down for 0.3 s each time), so we use **hysteresis**: two
different thresholds, one to go up to Li-Fi and a lower one to come back down, so the system does
not keep flipping between links.

**Words you will see.**
- **SNR (signal-to-noise ratio):** how strong the signal is compared to the noise, in dB. Higher is better.
- **BER (bit error rate):** the fraction of bits received wrongly. 10⁻³ means 1 wrong bit in 1000.
- **Throughput:** the data rate the user actually gets, in Mbps.
- **Outage:** the percentage of time the user gets almost nothing (less than 1 Mbps).
- **FOV (field of view):** how wide a cone the light sensor can "see".
- **Coverage:** the percentage of the room where the SNR is at least 15 dB (good enough to use).
- **Seed:** the starting number for the random-number generator. Same seed, same "random" walk.
- **95% confidence interval (CI):** "± x" means we are 95% sure the true average lies within that range.

All figures are in `results/figures/` and all numbers are in `results/results_summary.md`.

---

**E0: The original run (baseline).**
*Question:* does the new code give exactly the same answer as the original prototype?
*What we did:* we re-ran the prototype's single 300-second walk with seed 42 and drew the first 60 s.
*The plot:* the top panel shows the speed of each option over time. The bottom panel shows when the
light is blocked. You can see Li-Fi (orange) crash to zero during blockages while the hybrid (dashed)
drops to Wi-Fi.
*Takeaway:* the new code reproduces the prototype's numbers exactly, but this is only one random
walk, so we do not quote it as the final result.

**E1: Where in the room is the light signal strong?**
*Question:* how does the received light power and SNR change across the room?
*What we did:* we calculated them at about 15,000 points on the floor (desk height) for one LED in
the middle of the ceiling.
*The plot:* two colour maps; darker blue means stronger. Grey corners get no signal at all because
they are outside the sensor's field of view. The dashed circle marks 15 dB.
*Takeaway:* the signal is strong under the lamp and fades quickly towards the walls; only about 44%
of the room has a "good" (≥ 15 dB) signal.

**E2: How many bits arrive wrong?**
*Question:* how does the error rate grow as you walk away from the lamp, and is our BER formula correct?
*What we varied:* distance from the lamp, the modulation type (OOK sends 1 bit per pulse, 4-PAM 2 bits,
8-PAM 3 bits), and the SNR.
*The plot:* left, error rate vs distance for the three types; right, the formula (lines) vs a
simulation where we really send millions of random bits through noise (hollow markers).
*Takeaway:* sending more bits per pulse needs a much stronger signal for the same error rate, and
the simulation matches the formula (within 6%) for all three types, so we can trust it.

**E3: Which design choices give the best coverage?**
*Question:* how does coverage depend on how the lamp and sensor are built?
*What we varied:* one at a time: lamp height, how wide the lamp's beam is, how wide the sensor sees
(FOV), bandwidth, and lamp power.
*The plot:* the top row shows coverage; the bottom row shows SNR in the room corner (an orange x means
the corner gets no light at all).
*Takeaway:* more power helps a lot, a narrower sensor FOV (50°) concentrates light better, and a smaller
bandwidth gives better SNR but a lower top speed; every choice is a trade-off.

**E4: What happens when the light is blocked more often?**
*Question:* how do the three options cope as people block the light more and more?
*What we varied:* the chance of a blockage starting, from never up to 50% of the time.
*The plot:* left, average speed; right, outage. Error bars come from 30 random walks.
*Takeaway:* Li-Fi only gets much worse as blockage grows. The hybrid keeps outage much lower when
blockage is common, but when blockage is rare (below about 6% of the time) the hybrid's switching gaps
actually make it slightly worse than Li-Fi alone.

**E5: Which switching thresholds work best?**
*Question:* at what SNR should we move to Li-Fi (up threshold) and back to Wi-Fi (down threshold)?
*What we varied:* up threshold from 10 to 25 dB and down from 5 to 20 dB (down is always lower than up).
*The plot:* three colour maps (speed, outage, number of switches); the star is our chosen 15/10 dB.
*Takeaway:* lower thresholds give more speed because Li-Fi is already faster than Wi-Fi at low SNR,
while very high thresholds give fewer switches and lower outage; our 15/10 dB setting is a middle
compromise, not the best for speed (10/5 dB gives about 7% more).

**E6: The main result, with proper statistics.**
*Question:* averaged over many random walks, how do Wi-Fi only, Li-Fi only and the hybrid compare?
*What we did:* ran 30 independent 5-minute walks and reported the average with a 95% confidence interval.
*The plot:* bars are averages, black whiskers are the confidence interval, grey dots are the 30
individual runs.
*Takeaway:* the hybrid is about 3.3× faster than Wi-Fi only (398 vs 120 Mbps) and has less outage
than Li-Fi only (7.8% vs 9.7%), but it is a bit slower than Li-Fi only (398 vs 435 Mbps).
**These are the numbers to put in the final results table.**

**E7: What does hysteresis actually add?**
*Question:* is having two thresholds better than just one?
*What we compared:* Wi-Fi only, Li-Fi only, a single threshold at 10 dB, a single threshold at 15 dB,
our hysteresis 15/10 dB, and an "ideal" switch that always picks the faster link with no switching cost
(impossible in practice; it shows the best we could ever hope for).
*The plot:* bars for speed, outage and number of switches, each with a confidence interval.
*Takeaway:* hysteresis cuts the number of switches (80 vs 93 and 112) and gives the lowest outage of
the realistic hybrids, but a single 10 dB threshold is a bit faster, and all are below the ideal 446 Mbps.

**E8: What if we add more lamps? (stretch goal)**
*Question:* does a 2×2 grid of ceiling LEDs cover the room better than one?
*What we compared:* one 1 W LED, four LEDs at 1 W each, and four LEDs at 0.25 W each (the same total
power as one). The user connects to the strongest lamp. We ignore interference between lamps, which
makes this optimistic.
*The plot:* three SNR maps with the 15 dB edge dashed.
*Takeaway:* four lamps cover 100% of the room, but only because they use four times the power; with the
same total power, coverage drops to about 9%, because splitting light weakens every spot.

**S1: Do our simplifications change the conclusions?** (table in the summary, section 8)
We switched on, one at a time, some more realistic options that are off by default: a stricter Li-Fi
speed formula, Wi-Fi protocol overhead, and slowly changing Wi-Fi shadowing.
*Takeaway:* the stricter Li-Fi formula lowers Li-Fi speeds by about 16%, the Wi-Fi overhead lowers
Wi-Fi to 72 Mbps, and the shadowing option changes nothing, because our Wi-Fi model is always at its
maximum speed. The hybrid still beats Wi-Fi only by a wide margin in every case.

---

**Main limitations, in one line each.** We ignore light bouncing off walls. We ignore
interference between lamps. Our Wi-Fi model is very simple: it never slows down. Rates come from a
capped formula, not real hardware. The switching logic knows the SNR perfectly. See the README for
details.
