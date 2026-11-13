# Local retry comparison

The original pair remains preserved. Its naive run exceeded the configured generator-lag threshold. Observed lag clustered around Windows timer-sized increments; scheduler/clock quantization is a plausible contributor, not a proven sole cause.

The repeated pair requested a one-millisecond timer period for the driver and restored it afterwards. Both conditions used the same workload, seed, planned arrival schedule, development profile, fault settings and provider limits. Only retry policy differed within the pair. The application clock remained `time.monotonic_ns()`.

Both repeated runs passed the existing generator-lag check. The selected metrics and all earlier-run diagnostics are in `raw/metrics.json`; manifests and sampling audits retain actual decisions.

Fault settings and seeds match, but realized per-logical-request fault assignments are not guaranteed identical. The stand-in sampler consumes a seeded stream by provider-call sequence; retries and different call ordering change which logical request receives a draw. This is a limitation of the paired experiment, not evidence of an identical per-request fault schedule.

Naive retries produced a lower success rate in the observed repeated pair. They did not increase total provider calls. Duplicate slow work, abandonment, error outcomes and incomplete usage must be considered together; successful-only latency cannot establish a tail comparison with one successful naive response. These are small local development experiments, not Groq capacity or reliability estimates.
