# SpO2: turning the ring's R-ratio into a percentage

> This path uses **no model file** — just the small calibration constants below
> (read from the decompiled app). Unlike the activity/sleep/CVA runners, it needs
> none of Oura's proprietary `.pt` models (which are never committed — see
> [`docs/model-runners.md`](https://github.com/Th0rgal/open_health/blob/main/docs/model-runners.md) in open_health).

The ring's **`spo2_r_pi_event`** (tag `0x8b`) carries, per sample, an `r`
(ratio-of-ratios) and a perfusion index (`pi`) — **not** an SpO2 percentage. Two
paths convert it:

1. **Production nightly feature** — uses `API_SPO2_EVENT` (0x6f) /
   `API_SPO2_SMOOTHED_EVENT` (0x70), which are **percentages already computed in
   the ring firmware**; that R→% math lives in firmware. `0x6f` is decoded
   (`decode_spo2`). A Ring 3 Horizon writes it (one sample per second; values above 100
   occur, so clamp at 100) and does not write `0x8b`.
2. **"SpO2 Simple" path** — converts the `r` of `spo2_r_pi_event` in-app with a
   per-hardware quadratic. **This is the one we can reproduce** from our data.

## The formula

```
SpO2(%) = a·r² + b·r + c          clamped to [85, 100]
```

The polynomial runs inside `libecore` (`EcoreWrapper.nativeCalculateSpO2Simple`),
but the **coefficients are in the app** (`com/ouraring/oura/workitem/data/items/d.java`),
selected by ring hardware type:

| hardware | a | b | c |
| --- | --- | --- | --- |
| gen4 / oreo | −13.4 | −5.1 | 105.2 |
| cooper | −12.1 | −6.9 | 106.3 |

The Rust `oura_analysis::ported::spo2::spo2_simple(r, a, b, c)` evaluates `a + b·r + c·r²`
(constant term first) and clamps to 0..120, so it needs the table reversed: gen4 / oreo is
`spo2_simple(r, 105.2, -5.1, -13.4)`, and the caller has to clamp to 85..100. The Rust crate
holds no constants.

`pi` (perfusion index) is passed to the native algo as a quality input; there is no
in-app PI threshold. The daily value is clamped to integer **85–100**.

**Ring 5 caveat:** its exact hardware→coefficient mapping isn't pinned down in the
decompiled app (v7.18 predates Ring 5 in that table); the two sets differ by <1% on
our data, so the result is robust either way. Default is gen4/oreo.

## Running it

```
python tools/run_spo2.py [DB] [--hw gen4|cooper] [--night]
```

One night (26 492 samples, `--night`): mean **93.4%**, median 93%, p5 90%, min 85%
(clamp floor); ~5% of samples <90%. (A crude generic `110−25·r` gives ~91%, which is
wrong — use the calibrated quadratic.)
