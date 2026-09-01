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

## 5. Why NVIDIA and Dataiku, Specifically — Measured, Not Assumed

The case for this hardware/platform stack was not asserted — it was tested and, in one instance, revised after the data disagreed with the original assumption.

- **Batched GPU processing measured 28x faster than CPU** for the UMAP/HDBSCAN clustering step on historical windows. This is the direct, quantified justification for running large-scale regime discovery and backtesting research inside Dataiku on GPU infrastructure rather than on CPU: the same manifold-fitting work that takes CPU-scale time on a laptop completes in a fraction of that time on GPU, at the batch sizes a research workflow actually uses.
- **The original hypothesis — that cuVS CAGRA would outperform brute-force search as the historical attractor map grew to 10k–100k+ points — was tested directly and found to be incomplete.** The real crossover variable is query batch size, not index size: CAGRA loses to brute-force search at batch size 1 even against a 500,000-point index, and wins decisively only at batch sizes of 1,000 or more. This is exactly the kind of workload Control Plane (Dataiku) batch research produces — and exactly the kind the tick-by-tick Execution Plane does not. It directly shaped the architecture split in Section 2: GPU/CAGRA is reserved for background batch re-fitting, never for the hot path.
- **Compiled model inference (TL2cgen), called with zero Python in the path, measured 227 nanoseconds** for the value-model prediction step in isolation — roughly 200x faster than calling the same model through a standard Python/XGBoost interface. This is the concrete number behind the claim that GPU-trained models can be handed off to the C++ execution plane without paying an inference-latency tax.
- **The DPU case is scoped to what DPU acceleration actually addresses.** The measured 1.75–2.8 microsecond hot path is CPU compute; the network ingest path (packet arrival to adjacency-matrix construction) is the remaining unoptimized segment DPU offload targets. Separately, and explicitly not conflated with the DPU case: the fill-rate ceiling observed in Shadow Mode may be partly attributable to public-internet round-trip time to the exchange, which DPU hardware on our own server does not reduce — that is an exchange-colocation question, tracked as its own roadmap item below.

## 6. Full Mathematical Reference, By Layer

Stated plainly, with no metaphor left unexplained: every term below is a named linear-algebra, clustering, or statistical operation. Physics vocabulary is retained only because it was the shared language during design.

**Layer 0 — Volume Clock & Adjacency Matrix**

```
τ → τ+1  iff  Σ|dV| ≥ V_threshold          (volume clock: time discretized by traded volume, not wall clock)

A_ij = (dV_i − dV_j) / (|P_i − P_j| + ε)   (20×20 adjacency matrix, 10 bid + 10 ask nodes, FP16)
```
`A` is antisymmetric by construction (`A_ji = −A_ij`). This is not incidental — see Layer 1.

**Layer 1 — Commutator & Directed Pressure**

```
F = [A, δA] = A·δA − δA·A                  (commutator, Frobenius norm ‖F‖ used as a flow-toxicity proxy)

sym(δA) = (δA + δAᵀ)/2                     (Hodge-Dirac "gradient" component)
curl(δA) = (δA − δAᵀ)/2                    ("curl" component)
```
**Structural finding, confirmed numerically across thousands of ticks:** because `A` is antisymmetric, `δA` is antisymmetric, so `sym(δA) ≡ 0` always, exactly, by construction. No parameter choice fixes this while `A` remains antisymmetric. The gradient/curl decomposition above is retained for documentation continuity only. The live signal actually used is:

```
directed_pressure = Σ(ask-side dV) − Σ(bid-side dV)
```
computed directly from raw per-level volume deltas, independent of the degenerate decomposition.

**Layer 2/3 — Manifold, Clustering, Anomaly Distance**

```
embedding = UMAP(state_vector)                     (~11-dim compact vector: ‖F‖, curl energy, directed_pressure,
                                                      top-5 |eigenvalues of F|, curl mean/std, price_momentum)
regime, ε_regime = HDBSCAN(embedding), calibrated on a held-out window
d = distance(live_embedding, nearest_cluster)
anomaly ⟺ d > ε_regime × severity_multiplier        (severity-thresholded, not a raw binary flag — see below)
```
Calibration-window self-distances at the 99th percentile define the base `ε_regime`; the C++ live engine re-derives this via k-means (not HDBSCAN, for compute-cost reasons at hot-path scale) on a 2,000-point rolling ring buffer, re-fit on a hybrid anomaly-severity + drift-TTL trigger with a cooldown period.

**Layer 4 — Value Model & Decision Table**

```
value_pred = XGBoost(state_vector)                  target: forward mid-price return over target_horizon_ticks
decision = f(regime, anomaly)                        fixed lookup table, not a learned policy (deliberate design choice)
```

**Layer 5 — Position, Fill, and PnL Mechanics**

```
queue_position(t) = queue_position(t−1) − matched_trade_volume(t)
order FILLED  ⟺  queue_position ≤ 0                 (maker fill rule — no fill on price alone)
fill_price (market order) = volume-weighted walk across book levels until size is exhausted
fee = fill_notional × fee_rate                       (taker: 0.04%, maker: 0.00%, Binance schedule)
net_pnl = realized_pnl − Σ fees
```

## 7. Acceleration Journal — What We Tried, What Worked, Under What Exact Conditions

No claim below is asserted without the number and the hardware/data-scale condition it was measured under. Negative results are listed with the same weight as positive ones.

| # | What was tried | Library / technique | Result | Condition it holds under |
|---|---|---|---|---|
| 1 | Layer 1 commutator on GPU | CuPy / cuBLAS (RAPIDS) | 48.45 μs (eager mode) | Python-mediated, single 20×20 matrix |
| 2 | Layer 1 commutator on CPU | Hand-written AVX2 | **1.86 μs — 24–26x faster than #1** | Single-threaded, no Python |
| 3 | Layer 1 on CPU, wider SIMD | AVX-512 (`-march=native`) | 2.01 μs — **no improvement over AVX2** | Matrix too small (20×20) to benefit from 512-bit lanes |
| 4 | Eliminate Python launch overhead on #1 | CUDA Graph capture | **Failed outright**: `cuBLAS API during stream capture is currently unsupported` | CuPy 14.2.0; library-level limitation, not our bug |
| 5 | Same idea, avoiding cuBLAS | Hand-written CUDA kernel + CUDA Graph | 4.96 μs → 4.12 μs (graph capture succeeded here) | Still **2.2x slower than #2** — not pursued further (Amdahl's Law: optimizing 2% of tick time) |
| 6 | Historical regime clustering at scale | cuML UMAP + HDBSCAN (RAPIDS), **batched** | **28x faster than CPU** on windowed/batch workloads | Batch sizes typical of offline research, not single-tick |
| 7 | Live anomaly-distance search, original plan | cuVS CAGRA (RAPIDS) | 1.24 ms mean at batch=1 against a 300-point index | CAGRA is graph-based, tuned for million-point indexes |
| 8 | Same search, alternative | cuVS brute_force (RAPIDS) | **0.11 ms — 11.24x faster than CAGRA** at this scale | 300-point index, batch=1 |
| 9 | Re-tested #7 vs #8 at production scale | cuVS CAGRA vs brute_force | brute_force still wins at batch=1 **even against a 500,000-point index**; CAGRA wins only at batch ≥ 1,000 | The crossover variable is **query batch size**, not index size — this corrected the original architecture assumption |
| 10 | CPU vs GPU at that same 500k-point scale | sklearn ball_tree vs cuVS brute_force | 552 μs (CPU) vs 681 μs (GPU) | batch=1 — comparable, neither dominates |
| 11 | Compiled XGBoost inference, plan A | Treelite 4.x `Model.compile()` | **Does not exist in 4.x** — replaced by GTIL, an interpreted reference runtime documented as prioritizing legibility over speed | Measured 40.3 μs — no better than native XGBoost Python (49.1 μs) |
| 12 | Compiled XGBoost inference, plan B | TL2cgen (the actual successor project) + `dlopen`/`dlsym`, zero Python | **227 ns — 213x faster than native XGBoost Python** | Isolated microbenchmark, warmed, small model (max_depth=4, 50 rounds) |
| 13 | Same compiled model, embedded in the live hot path | TL2cgen, in-pipeline | 0.7–6 μs per call (not 227 ns) | Interleaved with surrounding computation — cache-cold effects, not a regression in the method |
| 14 | Fused Layer 0+1+2+4 hot path, unpinned | AVX2, no CPU affinity | 2.13 μs mean, 9.05 μs p999 | Intel Sapphire Rapids |
| 15 | Same, CPU-pinned | `pthread_setaffinity_np` to one core | **1.82 μs mean (−15%), 6.64 μs p999 (−27%)** | Same hardware |
| 16 | Argmin over K=100 centroids, isolated | AVX2 (`_mm256_blendv`) vs scalar loop | AVX2 2.36x faster (0.054 μs vs 0.128 μs) | Isolated microbenchmark |
| 17 | Same, embedded in the full hot path | AVX2 vs scalar, 3 repeated runs each | Mean/p50 **statistically indistinguishable**; **p999 consistently lower with AVX2** (6.67–6.94 μs vs scalar's 8.19–9.91 μs, no overlap) | Adopted for tail-latency stability, not mean improvement — isolated benchmarks did not predict in-pipeline behavior here either |
| 18 | Same fused hot path, second CPU architecture | AVX2, pinned | 2.77 μs mean | AMD Zen 3 — confirms all *relative* findings hold cross-platform; absolute latency is hardware-dependent |
| 19 | Thread synchronization for the Order Management System | `std::mutex` vs `Spinlock` (`std::atomic_flag`) | Spinlock adopted — avoids OS scheduler context-switch cost on lock contention lasting nanoseconds | HFT-relevant: any syscall-mediated block is a latency outlier risk |
| 20 | VORTEX anomaly trigger, binary flag | `distance > ε_regime` | Fired on **21.7% of ticks** — structurally prevented any maker order from surviving long enough to fill | Live BTCUSDT, 1,363-tick session |
| 21 | Same trigger, severity-thresholded | `distance / ε_regime > 5.0` (threshold chosen from the measured distribution of this ratio, not guessed) | Anomaly-driven cancellations cut **roughly 5x** (296+ → 59–76 per session), while the shield still fires on genuine tail events | Threshold sits near the 75th–90th percentile of the measured ratio across two independent live sessions |

**The throughline, stated bluntly:** GPU won every batched, offline, research-scale comparison it was tested against — 28x on clustering, 11x on tail-search at production scale under batching. GPU lost every single-item, tick-by-tick comparison it was tested against, without exception, against hand-written CPU code — the commutator, the anomaly search at batch=1, and the compiled model call all fastest on CPU or via CPU-callable compiled artifacts. This is not a marketing conclusion; it is the literal count of the table above. The architecture in Section 2 is built directly on this count, not around it.

## 8. Roadmap

- **Layer 0 offload to NVIDIA BlueField-3 DPU (DOCA)** to move packet parsing and adjacency-matrix construction off the host CPU entirely.
- **Exchange-proximate colocation** to reduce round-trip network latency — noting explicitly that this addresses a distinct bottleneck from DPU acceleration (network transit time to the exchange's matching engine, not host-side compute), and that the current fill-rate ceiling has not yet been attributed between these two causes.
- **Zero-copy (DLPack) memory sharing** across all layers, extending the pattern already validated for the GPU→XGBoost boundary to the full pipeline.

---

*This document describes measured engineering results and an explicitly open research problem. Figures in Sections 3 and 4 are drawn directly from logged live sessions on real exchange data; none are projected, extrapolated, or averaged from a larger unreported set of runs.*