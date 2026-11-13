# Real evaluation: decision before dispatch

I rechecked the runner, policy, durable real-provider ledger and supplied screenshots. The full A/B/C evaluation cannot pass the existing conservative preflight. I have not raised caps or sent another Groq request.

| Setting, 18 questions | Initial draft + verifier reservation per question | Initial pairs for 18 | Including configured repair and shared retry |
|---|---:|---:|---:|
| A | 4,670 | 84,060 | 255,420 |
| B | 5,070 | 91,260 | 277,020 |
| C | 4,870 | 87,660 | 266,220 |
| All 54 | — | 262,980 | 798,660 |

These are reservation estimates, not measured usage. Completion budgets include reasoning. A repair includes the rejected draft and complaint allowance; a single retry is shared across the logical request. Each setting allows at most 90 attempts across its 18 questions. Each actual dispatch still reserves its serialized prompt and bounded completion through the normal control plane and ledger.

The unchanged session caps are 60,000 estimated tokens and 150 calls. Each complete setting exceeds the token cap even before repair/retry allowances. The screenshot establishes a 200,000-token daily account limit; it does not establish available balance. The full initial-pair estimate exceeds that entire daily limit. The runner also requires the planned batch to fit 70% of fresh verified remaining daily quota. Pacing can address the 8,000 TPM limit, but cannot increase the session or daily budgets.

The latest instruction requires stopping before dispatch if the full planned evaluation does not fit. Splitting invocations, resetting sessions, reducing estimates, removing validation or silently lowering safety margins would not establish that the full evaluation fits. A separately authorized partial study would remain incomplete and would need fresh quota evidence, explicit question selection, pacing, the shared ledger and preserved campaign accounting. It is not a completed A/B/C comparison.

The genuine Limits screenshot now supports 30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD. The Usage screenshot is an aggregate view with a warning about reporting delay. Its filename and filesystem modification time are not verified capture timestamps. Historical remaining headers belong to their recorded requests. The connected browser's direct Groq console attempt returned “Access denied. Please check your network settings”; no fresh balance could be verified there.

The real ledger still contains two attempts and 2,084 reported tokens (1,780 input and 304 output). Its latest attempt is 2026-11-13T19:01:43.139883+00:00. This establishes no intervening usage recorded by Relay; it cannot rule out another client using the organization. Cached-token uncertainty retains a larger conservative guard charge, rather than assuming zero.

Dataset approval remains explicitly pending. I have not represented assistant source checks as unaided candidate review. Six blank stand-in timeouts remain ungraded; assigning them answer-quality grades would invent an answer that was never delivered. Calibration remains one draft observation and zero verifier observations. Stand-in reviews and development loads cannot fill those real-provider evidence gaps.

I fixed an independent runner defect: a limited subset was executed after a preflight that counted all 18 questions. The manifest and preflight now use the same selected question list. A subset is still labelled incomplete. Malformed, non-finite or timezone-free quota evidence fails closed, and an unreadable evidence file leaves a blocked manifest. All regression tests use offline dispatch-boundary stubs.

The hiring email supplied in chat and the opened form agree on the submission destination. The form requires Google sign-in. No form submission or confirmation has been observed. The final package remains suitable for review as an honest partial submission; it does not satisfy the missing experimental requirements.
