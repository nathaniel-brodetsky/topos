# TOPOS v0.9 Beta
### Infrastructure for Quantum Market Microstructure

**Status: Beta.** The execution infrastructure is validated. The trading strategy is a reference implementation, not a proven profitable system. See Section 4 before drawing conclusions about alpha.

---

## 1. Vision

Classical CPU-bound HFT has run into a hard physical ceiling: the Python GIL serializes what should be parallel tick processing, kernel-mediated I/O adds unavoidable latency at every network and memory boundary, and naive taker-side execution pays away any short-horizon edge in fees before it can compound.

TOPOS does not try to out-optimize this ceiling from inside it. Python is used exclusively as a control-plane conductor — orchestrating GPU/VRAM resident computation (UMAP, HDBSCAN, XGBoost via RAPIDS) — while the tick-by-tick execution plane is a dependency-free C++ hot path with no interpreter in the critical loop. Concepts borrowed from gauge theory and topology (commutators, manifold clustering, gauge-invariant pressure) are used here as an engineering vocabulary for concrete linear-algebra and density-clustering operations — not as physical claims about markets. This framing was a deliberate design choice from the start of the project and is maintained throughout this codebase.

## 2. Architecture — The Five Layers

```
Layer 0   Ingestion & Volume Clock
          Market time is decoupled from wall-clock time; the system advances
          on accumulated traded volume, not elapsed seconds. Raw depth and
          trade streams are parsed into a fixed 20-node (10 bid + 10 ask)
          adjacency matrix.

Layer 1   Gauge Dynamics / Directed Pressure
          The commutator [A, δA] of the adjacency matrix is used as an
          empirical proxy for order-flow toxicity, calibrated against
          historical data, not asserted as a physical law. A structural
          degeneracy (the symmetric component is provably zero given an
          antisymmetric adjacency matrix) was identified and documented;
          directed_pressure is computed independently from raw volume
          deltas as the working signal.

Layer 2/3 Adaptive Manifold (UMAP + HDBSCAN)
          Live state vectors are projected onto a self-calibrating manifold
          and clustered into regimes (EQUILIBRIUM / IMPULSE / TRANSITIONAL /
          anomaly). A hybrid trigger (anomaly severity + drift TTL) with a
          Topological Cooldown period governs when the manifold is
          re-fitted, preventing the feedback loop where every re-fit
          destabilizes the very state used to trigger the next one.

Layer 4/5 Decision & Execution (XGBoost + OMS)
          A compiled (TL2cgen) XGBoost model and a fixed decision table
          drive a C++ Order Management System: a pessimistic Shadow
          Matcher simulating realistic queue-position fills and exchange
          fee schedules, synchronized across threads via spinlocks (not
          OS mutexes) to keep the hot path free of kernel scheduling
          jitter.
```

## 3. Current Engineering Achievements

These are measured results, verified through repeated runs on live exchange data — not projections.

- **Binance's asynchronous multi-stream WebSocket migration was diagnosed and solved.** Depth and trade streams silently drop when mixed under an unrouted or mismatched-category URL; the correct architecture uses two independent WebSocket connections on their respective routed paths, verified running concurrently without conflict.
- **A pessimistic, queue-position-aware order matching simulator (Shadow Matcher)** was built and unit-tested: limit orders only fill when real aggregated trade volume exhausts the queue ahead of them; market orders walk the book across levels with honest slippage; Binance's fee schedule is applied exactly.
- **The Zero Taker Fee Rule is enforced architecturally, not by convention.** `place_market_order` is never called in the current execution logic. Across 5 independent 150-second live sessions on real Binance BTCUSDT data, the maker fill ratio was 100% and total fees paid were 0.000000 in every session.
- **False-positive anomaly cancellations were reduced substantially through severity filtering.** The original binary anomaly flag (`distance > epsilon`) fired on ~20% of ticks, making it structurally impossible for any resting maker order to survive long enough to fill. Replacing this with a severity ratio threshold (calibrated against the measured distribution of the ratio across live sessions) cut anomaly-driven cancellations by roughly 5x while preserving the shield's function during genuine tail events.
- **The full C++ hot path (Layer 0 through Layer 4 decision) runs in 1.75–2.8 microseconds** (measured on both Intel Sapphire Rapids and AMD Zen 3, with expected hardware-dependent variation), verified via CPU-pinned, warmed-up, repeated benchmark runs — not a single favorable measurement.

## 4. The Alpha Sandbox — Strategy as an Open Problem

This is the section that should not be skimmed. The current trading logic (adaptive pegged quoting, regime-conditioned routing across FLAT/IMPULSE/VORTEX) is a **reference implementation**, built to prove that the execution engine can survive and operate without bleeding fees — not a validated source of profit.

**What was measured, in full, across 5 independent 150-second live sessions:**

| Session | Fills | Net PnL |
|---|---|---|
| 1 | 0 | 0.000 |
| 2 | 2 | +0.047 |
| 3 | 4 | −0.057 |
| 4 | 7 | +0.171 |
| 5 | 6 | +0.155 |
| **Total** | **19** | **+0.316** |

Maker ratio was 100% and fees were zero in every session. Three of five sessions were net positive, one was net negative, one was flat. This is read honestly as: **the engine demonstrated stable survivability — zero fee bleed, correct anomaly avoidance, no catastrophic loss — on a sample too small to constitute a statistically validated profitable edge.** Nineteen fills cannot distinguish a real, small positive expectancy from chance.

What this hands to a quant researcher is a genuinely rare asset in this space: an execution engine that has already solved the two problems that usually kill short-horizon signals before they can be evaluated fairly — taker fee bleed and toxic-flow blowups. The open task is squarely on the alpha side: improving the XGBoost value model (currently trained on synthetic proxy features, not real forward returns) and the regime-to-action routing logic, using an engine that will not itself destroy a real edge through execution cost. That is a fundamentally different, more tractable starting point than building both the signal and the execution plane from zero.

## 5. Roadmap

- **Layer 0 offload to NVIDIA BlueField-3 DPU (DOCA)** to move packet parsing and adjacency-matrix construction off the host CPU entirely.
- **Exchange-proximate colocation** to reduce round-trip network latency — noting explicitly that this addresses a distinct bottleneck from DPU acceleration (network transit time to the exchange's matching engine, not host-side compute), and that the current fill-rate ceiling has not yet been attributed between these two causes.
- **Zero-copy (DLPack) memory sharing** across all layers, extending the pattern already validated for the GPU→XGBoost boundary to the full pipeline.

---

*This document describes measured engineering results and an explicitly open research problem. Figures in Sections 3 and 4 are drawn directly from logged live sessions on real exchange data; none are projected, extrapolated, or averaged from a larger unreported set of runs.*
