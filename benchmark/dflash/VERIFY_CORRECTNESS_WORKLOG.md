# DFlash Verify-Kernel Correctness Worklog

## Status

Definition and design phase complete for the approved initial profile.

A strict source review has been completed at the pinned repository/model
revisions below. All explicitly listed blocking contract decisions have been
approved.

No correctness harness, correctness experiment, verify-kernel change, or
verify-kernel optimization has been started under this worklog. A subsequent
implementation phase must conform to this contract and version any deviation.

Repository state at the start of this worklog:

- Branch: `dcw02/kimi-k3-dflash`
- Commit: `7125fa3bee69b3e71974b3ab806eb2aa5912c6c8`
- Date: 2026-08-02

Anything in the existing Git stash is reference material only. It is not the
specification, an oracle, or an implementation to restore. This investigation
will be reconstructed from first principles. After the design is settled, the
stash may be inspected for useful regression cases or implementation ideas.

## Strict-Review Basis

This document was reviewed against:

- SGLang commit `7125fa3bee69b3e71974b3ab806eb2aa5912c6c8`;
- the benchmark's pinned `moonshotai/Kimi-K3` model revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`;
- the model's checked-in remote implementation,
  `modeling_kimi_linear.py`, at that revision;
- the current DFlash worker, accept/bonus kernel, scheduler result processing,
  KDA backend and kernels, Mamba state commit, and ReplaySSM fold code.

The code is evidence about the deployed implementation, not the definition of
the analytic target. Comments and tests that call Triton a “reference” or
ReplaySSM a “bitwise clone” record implementation intent only; they cannot make
either backend the oracle for this investigation.

The principal implementation evidence at the pinned commit is:

- `python/sglang/srt/speculative/dflash_worker_v2.py`,
  `python/sglang/srt/speculative/dflash_utils.py`, and
  `python/sglang/kernels/ops/speculative/dflash.py` for row layout, greedy
  selection, prefix acceptance, emitted tokens, and commit lengths;
- `python/sglang/srt/layers/logits_processor.py` and
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
  for valid-vocabulary scores and user-visible result processing;
- `python/sglang/srt/layers/attention/linear/kda_backend.py`,
  `python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py`, and
  `python/sglang/srt/models/kimi_k3.py` for direct/fallback dispatch, the
  fused KDA path, and gated RMSNorm;
- `python/sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py` and
  `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` for
  ReplaySSM folding and logical state selection; and
- the pinned model's `modeling_kimi_linear.py` and `config.json` for the
  architecture-level equations and constants.

## Current Scope

The initial terminology applies to concurrency-one, strict greedy speculative
decoding. Sampling requires a different acceptance predicate and is outside
the current definition. The first audit profile must also exclude:

- grammar constraints;
- custom logit processors, repetition/frequency/presence penalties, and
  request logit bias;
- simulated-acceptance environment knobs; and
- any request for which the eligible vocabulary or post-adjustment greedy
  scores cannot be reconstructed exactly.

Those features can be added later by extending the target decision contract.
EOS, stop strings, and maximum-token limits do not change the verifier's
internal accept/commit decision, but they can truncate what becomes
user-visible. Reports must therefore distinguish verifier-committed tokens from
user-visible tokens.

A qualifying trace must prove that `sampling_info.is_all_greedy` was true and
that DFlash took its direct `torch.argmax` verification branch. It must record
temperature/top-k/top-p fields, even when the greedy branch makes them
operationally irrelevant, and must prove that no sampling fallback or simulated
acceptance path ran.

Let \(P\) be the target state/cache prefix at block entry and let \(s\) be the
previously emitted, not-yet-consumed seed in `candidates[:, 0]`. For proposal
\(p_i\), the target decision row is evaluated on

\[
P \mathbin\Vert s \mathbin\Vert p_1 \mathbin\Vert \cdots
\mathbin\Vert p_{i-1}.
\]

All compared target computations must therefore use the same:

- committed prefix;
- seed token;
- accepted proposal prefix \(p_1, \ldots, p_{i-1}\);
- target weights;
- model semantics; and
- explicitly defined numerical contract.

The final item is intentionally unresolved. No existing decode, prefill, or
verify backend is assumed to be an unquestioned oracle.

## Provisional Decision Definitions

Let \(E_i\) be the eligible vocabulary after all in-scope masks and let
\(\ell_i(t)\) be the final target score for token \(t\). The strict greedy target
token is

\[
G_i = \min \operatorname*{arg\,max}_{t \in E_i} \ell_i(t).
\]

The minimum implements SGLang's first-index `torch.argmax` tie rule on the
full, valid vocabulary. Padded vocabulary entries are excluded. A trace with an
eligible nonfinite score, no eligible token, missing score adjustment, or
unreconstructable vocabulary layout is **invalid/unclassified**, not
numerically indeterminate. An out-of-vocabulary proposal is invalid; an
in-vocabulary proposal excluded from \(E_i\) is a stable reject by the mask
without a numerical score comparison.

For each competitor \(j\in E_i\setminus\{p_i\}\), the analysis should directly
enclose the pairwise difference

\[
d_{p,j} = \ell_i(p_i) - \ell_i(j)
\]

with a certified interval \([d_{p,j}^{lo},d_{p,j}^{hi}]\). Then:

- **Stable accept:** for every eligible \(j < p_i\),
  \(d_{p,j}^{lo} > 0\), and for every eligible \(j > p_i\),
  \(d_{p,j}^{lo} \geq 0\). The proposal is certifiably \(G_i\).
- **Stable reject:** there exists an eligible \(j < p_i\) with
  \(d_{p,j}^{hi} \leq 0\), or an eligible \(j > p_i\) with
  \(d_{p,j}^{hi} < 0\). The proposal is certifiably not \(G_i\).
- **Indeterminate:** the evidence proves neither condition. This is proof
  incompleteness, not a runtime decision and not automatically a genuine
  “tolerance zone.”
- **False accept:** target verification accepts a stably rejected proposal.
- **False reject:** target verification rejects a stably accepted proposal.
- **Correct accept:** target verification accepts a stably accepted proposal.
- **Correct reject:** target verification rejects a stably rejected proposal.

The inclusive tie branches require a certified sign or exact-equality result.
Merely increasing interval precision does not in general prove that a
transcendental expression is exactly zero. If every enclosure continues to
straddle zero and no independent symbolic/equality certificate is available,
the event remains indeterminate even though the semantic tie rule itself is
fully specified.

The same tie-aware test, applied to any token \(q\), certifies whether \(q=G_i\).
Proposal acceptance needs only certify whether \(p_i=G_i\). Certifying a
replacement or bonus token additionally requires certifying the identity of
\(G_i\).

The approved classification table is:

| Certified target fate | Verifier accepts | Verifier rejects |
| --- | --- | --- |
| Stable accept | Correct accept | False reject |
| Stable reject | False accept | Correct reject |
| Indeterminate | Indeterminate | Indeterminate |
| Invalid/unclassified | Invalid/unclassified | Invalid/unclassified |

“Correct” in this table refers only to the acceptance decision. Token emission
and state transition are separate axes. An indeterminate result is neither a
pass nor a proven failure. It must be reported with a reason code and must not
be silently assigned to the category that makes a backend look better. An
invalid trace supplies no numerical evidence at all.

## Reachability

Reachability has two independent dimensions:

- A position is **runtime-reachable** when the observed verifier accepted every
  preceding proposal in the block. Once it rejects, later target rows still
  exist because the block was evaluated under a teacher-forced proposal chain,
  but they are counterfactual to the runtime.
- A position is **reference-aligned** when the emitted prefix is still
  certified equal to the strict target trajectory. This requires the seed and
  every earlier emitted replacement/bonus to have certified winner identity,
  every emitted proposal on that trajectory to be a stable target accept,
  correct state transitions, and no accepted-indeterminate or other unresolved
  event. Within one runtime-reachable block, all preceding proposals were
  accepted, so they must be certified accepts. Across a block boundary,
  acceptance-decision correctness is not itself the alignment criterion: a
  structural false reject can preserve alignment if its emitted replacement is
  the same certified target token and the pre-emission state is correct.

A position after a false accept can be runtime-reachable while no longer being
reference-aligned. It may be analyzed conditionally on that already-divergent
prefix, but it must not inflate pre-divergence false-accept or false-reject
counts. A position after an accepted-indeterminate event has unresolved
reference alignment and must be tagged separately.

Primary trajectory reporting should stop causal attribution at the first
certified failure or unresolved alignment event. Counterfactual and
post-divergence rows remain useful kernel coverage, but must be reported in
separate strata.

## Distinct Correctness Questions

The investigation must keep these questions separate:

1. **Numerical fidelity:** how closely does the verify kernel approximate the
   specified target computation?
2. **Acceptance-decision correctness:** does the verifier make the certified
   accept/reject decision for the proposal?
3. **Emitted-token correctness:** when verification rejects, does the runtime
   emit the certified target token?
4. **State-transition correctness:** after each possible commit frontier, are
   the recurrent state, convolution history, and other committed target caches
   the certified state for the consumed prefix?
5. **Draft-state materialization:** did the committed target hidden prefix
   populate draft KV at the intended rows? This can change later proposals and
   performance without itself changing target semantics.
6. **End-to-end generation fate:** where does the speculative trajectory first
   diverge, and which decision caused the divergence?

A false accept necessarily places a target-invalid proposal in the verifier's
raw emitted run. EOS, stop-string, or maximum-token postprocessing may truncate
user-visible output, which is why those runs are reported separately; it does
not make the verification decision correct. Grammar is different: it changes
the eligible set before `torch.argmax` and requires the extended decision
contract excluded from the first profile. A false reject is only a pure
performance loss if the rejection path independently recovers and emits the
certified target token and leaves the target cache at the certified
pre-emission frontier. Otherwise, a false reject may also alter generation.
Therefore acceptance fate, emitted-token fate, and state-transition fate must
be tracked independently.

A state error can be latent: the current emitted token may be correct while the
corrupted state changes a later decision. Agreement at the current token cannot
discharge a persistent-state mismatch.

Numerical differences that preserve a certified decision are not false accepts
or false rejects. Conversely, small-looking tensor errors are still important
if they cross a certified decision boundary.

## Non-Speculative and Speculative Generation Trajectories

Non-speculative target generation and speculative target generation are
different numerical execution paths. In exact arithmetic, a correct greedy
speculative implementation of the same target semantics would emit the same
tokens as non-speculative generation. That equivalence is not guaranteed for
the actual kernels because their numerical computations can differ.

The observed non-speculative trajectory is not automatically the certified
target trajectory, and the observed speculative trajectory is not automatically
incorrect merely because the two trajectories differ. A divergence must be
classified against the independently defined target contract.

In particular, zero false accepts does **not** imply that speculative and
non-speculative generation remain token-identical. A false reject can cause
divergence:

1. The proposal token is a stable target accept.
2. The verifier rejects that proposal.
3. The rejection path emits a different token.
4. The speculative prefix and subsequent generation diverge.

This is a **divergent false reject**. A false reject that ultimately emits the
same certified target token is a **non-divergent false reject**: it is still an
acceptance-decision error and a performance loss, but it does not cause token
divergence at that position.

There are also possible divergence mechanisms with no false accept and no false
reject:

- The target stably rejects the proposal and the verifier correctly rejects it,
  but the verifier emits the wrong competing token.
- Every proposal in a block is correctly accepted, but the verifier's bonus
  token differs from the certified next target token.
- Acceptance and the currently emitted tokens are correct, but the committed
  recurrent, convolution, KV, or hidden-state cache is wrong and changes a
  future decision.
- The relevant target decision is indeterminate, so two numerical paths choose
  different sides of an uncertified boundary.
- The non-speculative runtime itself differs from the certified target decision.

Therefore the investigation must compare three distinct objects:

1. the certified target decision under the agreed numerical contract;
2. the observed non-speculative runtime decision and emitted token; and
3. the observed speculative verifier decision and emitted token.

For each request, the eventual design should identify the first token where the
observed trajectories differ, then classify the responsible event as a false
accept, divergent false reject, incorrect rejection token, incorrect bonus
token, latent state-transition failure, indeterminate decision,
non-speculative deviation, or another explicitly defined category. Mere
spec-versus-non-spec disagreement is evidence of divergence, not by itself
evidence of which path is correct.

## Current SGLang Greedy DFlash Semantics

The current implementation uses the following shifted-token layout:

- `candidates[:, 0]` is the previously emitted target token that seeds this
  verify block. It sits immediately after the committed target-cache prefix: it
  has been emitted, but has not yet been consumed into target state.
- `candidates[:, 1:]` are draft proposals.
- `target_predict[:, i]` is the greedy target-verify prediction produced after
  processing candidate position \(i\). In particular,
  `target_predict[k]` is the next-token decision on
  \(P\Vert s\Vert p_1\Vert\cdots\Vert p_k\).

The implementation accepts the longest prefix satisfying

\[
\text{candidates}[i + 1] = \text{target\_predict}[i].
\]

If \(k\) draft proposals match, it:

1. emits those \(k\) proposal tokens;
2. emits `target_predict[k]` as the target bonus/replacement token;
3. commits the \(k + 1\) target-verify inputs
   `candidates[0:k+1]`; and
4. uses that bonus token to seed the next block.

Thus two equal-length token runs must not be conflated:

\[
\text{emitted} = [p_1,\ldots,p_k,\text{target\_predict}[k]],
\]

\[
\text{consumed into target state} =
[\text{seed},p_1,\ldots,p_k].
\]

The emitted bonus/replacement token is not yet represented in the target state.
It is consumed as `candidates[:, 0]` in the next block. The correct committed
state frontier is therefore the state after the final committed verify input,
immediately before the emitted bonus/replacement token.

Here **commit** means logical visibility after rollback/selection. The target
forward evaluates all rows in the verify block and may physically write
intermediate outputs or scratch for all of them. Only the first \(k+1\)
verify-input positions become logically visible to sequence length and target
state. The audit must distinguish physical writes from logical commit, and
must check target recurrent/convolution state separately from draft-KV
materialization; equal prefix lengths do not make those caches the same object.

Accordingly, SGLang's returned `accept_lens` is \(k+1\): both the number of
tokens emitted this iteration and the number of verify-input positions
committed, even though those two runs contain different endpoint tokens.
The conventional statement that it “includes the bonus” refers to the emitted
run; it does not mean the bonus has already entered target state. A reported
`accept_lens == 1` means zero accepted draft proposals, one target-produced
output token, and one consumed seed input. The internal draft-only acceptance
length is zero.

The returned `next_token_ids` buffer still has width \(D\) and can contain an
unused proposal suffix. Only `out_tokens[:accept_lens]` for each request is the
raw verifier-emitted run; buffer contents after that slice are not decisions or
emitted tokens.

This behavior is implemented by:

- `compute_dflash_correct_drafts_and_bonus` in
  `python/sglang/srt/speculative/dflash_utils.py`;
- `_dflash_accept_bonus_contig_kernel` in
  `python/sglang/kernels/ops/speculative/dflash.py`; and
- `DFlashWorkerV2.forward_batch_generation` in
  `python/sglang/srt/speculative/dflash_worker_v2.py`.

There is no independent target recomputation after a greedy mismatch. The same
target-verify row argmax that causes the mismatch is emitted as the bonus token.
If all proposals match, the final row `target_predict[D-1]` is emitted as the
bonus even though that row never participates in proposal equality. Its winner
identity therefore needs an independent certificate.
The greedy DFlash path calls `torch.argmax` directly rather than the normal
sampler's invalid-value sanitation path, so a nonfinite eligible score is a
structural audit failure, not a value to normalize away in the oracle.
This creates two materially different cases:

1. **Accept-prefix truncation error:** the supplied target row predicts the
   proposal, but the accept-prefix logic stops early. The emitted bonus at that
   position is the same token as the proposal. The token is deliberately left
   as the next block's unconsumed seed. This is a performance-only failure only
   if that token is independently certified as \(G_i\) and the state after the
   preceding consumed input is correct.
2. **Certified target-row false reject:** the certified target accepts the
   proposal, but the target-verify row chooses a different argmax. SGLang emits
   that different argmax immediately. Under a strict certified-target contract,
   this is both a missed draft acceptance and an emitted-token divergence.

The current Python and Triton accept-prefix implementations perform direct token
equality followed by a prefix scan. The verify-kernel investigation is primarily
concerned with the second case: numerical differences in target verification
can change `target_predict`, rather than the small prefix scanner independently
discarding a matching token.

Relative to the verifier's own runtime logits, SGLang never emits the rejected
draft proposal: it emits a token selected by the target-verify forward. Whether
that is sufficient to call the output “correct” is a target-contract decision.
It is sufficient under verifier-relative runtime semantics, but not by itself
under a contract that independently certifies the intended target computation.
The fact that a token came from the target-verify call cannot certify the call
whose numerical correctness is itself under audit.

## Correctness, Performance, and Indeterminate Outcomes

At the proposal acceptance gate:

- A runtime-reachable, reference-aligned false accept is a hard correctness
  failure because the runtime permits a proposal that the target stably
  rejects. Conditional false accepts after an earlier divergence remain kernel
  evidence but are not additional first-trajectory failures.
- A false reject is a missed speculative opportunity. It lowers the effective
  acceptance length and therefore degrades the performance realized from the
  draft model.

Calling a false reject **performance-only** requires two additional facts:

1. the rejection path emits the certified target token; and
2. the target state after the final consumed verify input is the certified
   state immediately before that emitted token.

If either condition fails, the acceptance decision still contains a false
reject, but the request also has an emitted-token or state-correctness failure.
This distinction prevents a generation divergence from being hidden inside a
performance-only category.

Indeterminate is a property of the certification evidence, not a third runtime
decision. The runtime still either accepts or rejects. Reports must therefore
distinguish:

- **accepted indeterminate:** the verifier accepts a proposal whose certified
  target fate is unresolved; and
- **rejected indeterminate:** the verifier rejects a proposal whose certified
  target fate is unresolved.

Neither outcome may be counted as a correct accept/reject or as a proven false
accept/reject. Under a strict correctness policy, an accepted indeterminate is
a possible false accept and a rejected indeterminate is a possible false
reject.

The performance objective should be to maximize **certified accepts**, subject
to certified absence of false accepts on the claimed scope and correct
replacement-token/state recovery. Merely observing zero certified false accepts
is insufficient when accepted-indeterminate, invalid, missing, or unaudited
events remain. Indeterminate cases should be reduced by:

- improving the verifier's numerical fidelity;
- tightening the independently justified uncertainty bounds; or
- collecting enough intermediate evidence to certify the final decision.

The classification boundary must not be moved after observing outcomes merely
to favor the draft. If the intended product semantics permit any token within a
defined numerical-equivalence set, that must be adopted explicitly as part of
the target contract. Such tokens would then be judged against that acceptance
set rather than silently relabelled from indeterminate to accepted. This policy
would permit some spec-versus-non-spec divergence by definition and must remain
distinct from strict greedy equivalence.

Every indeterminate result should carry a reason such as insufficient working
precision, loose downstream propagation, uncertified input-state provenance,
or a resource cap. Missing data needed to reconstruct the decision makes the
trace invalid/unclassified instead. With exact represented inputs and a
deterministic real-valued contract, arbitrary precision should resolve every
nonzero comparison; an exact tie is resolved by the declared token-order rule.

## Requirements for “Stable” and “Indeterminate”

The numerical uncertainty must be justified by the intended computation. A
fixed `atol`/`rtol`, agreement between two kernels, or agreement with non-spec
generation is not sufficient by itself.

Before implementation, the design must specify:

- the exact real-valued operator, exact represented inputs, and genuine
  external dtype/state boundaries;
- how uncertainty is derived or bounded;
- how argmax ties and deterministic tie-breaking are handled;
- whether certification is performed at kernel outputs, final logits, or both;
- how uncertainty composes through downstream target layers; and
- what evidence is required to call a target decision stable.

If a scalar shorthand is eventually used, define the strict proposal margin
\(m_i=\ell_i(p_i)-\max_{j\in E_i,\;j\ne p_i}\ell_i(j)\), let
\(\widehat m_i\) approximate it,
and independently prove
\(|m_i-\widehat m_i|\leq\tau_i\). Then
\(\widehat m_i-\tau_i>0\) certifies a strict accept margin and
\(\widehat m_i+\tau_i<0\) certifies reject; otherwise this shortcut is
indeterminate and the tie-aware pairwise rule above remains normative.
Asymmetric bounds should be retained when available. Each bound must come from
the error analysis and must not be selected after observing verifier outcomes.
For a singleton eligible set containing \(p_i\), accept directly rather than
forming the empty maximum (equivalently define \(\max\varnothing=-\infty\)).

## Reference-First Sequencing

The analytic reference must be specified and frozen before implementation work
on the CuTe verify kernel resumes. The KDA operator reference is
backend-independent; CuTe, Triton, and any future KDA kernel are replaceable
implementations judged against it. The initial final-decision claim may compose
that operator with the explicitly implementation-relative operational hybrid
defined below. Reports must name both layers of that contract rather than call
the whole hybrid backend-independent.

The intended order is:

1. Define the target operator's exact represented inputs, real-valued
   equations, genuine external storage/state projections, and tie-breaking
   rule. Do not import an implementation's internal cast or reduction order.
2. Specify an independent high-precision or interval evaluation of that
   operator. BF16/FP32 inputs are treated as their exact represented values,
   while internal real variables are not rounded merely because a backend
   happens to materialize them.
3. Establish how the reference kernel output is propagated to the final target
   logits and proposal-versus-competitor margin. A layer-local reference alone
   is insufficient to classify false accepts or false rejects at the token
   decision.
4. Define the reference's own uncertainty interval and the evidence required to
   trust it.
5. For a trace audit, compare the candidate's exact observed output/state bits
   with the analytic enclosures. For a universal kernel proof, additionally
   model or verify the candidate program's accumulation order, approximate
   transcendental instructions, cast points, indexing, and synchronization over
   the full legal input domain.
6. Evaluate every optimization candidate against the unchanged analytic
   reference and decision criteria.

An optimization may change scheduling, tiling, reduction order, state staging,
or approximation instructions without changing the reference. If an
optimization intentionally changes the target equations or external state
contract, it is a semantic-contract proposal, not an ordinary optimization,
and requires explicit review before the reference is changed.

“Zero false accepts” must always mean zero certified false accepts relative to
this fixed reference in the audited population. Agreement with the previous
CuTe kernel, the Triton kernel, or observed non-speculative output is supporting
evidence only and cannot replace the analytic reference.

### Claim Strength

The eventual report must distinguish three claims:

1. **Observed count:** no certified false accept was found among the events that
   could be classified. This weak claim is compatible with every accepted event
   being indeterminate.
2. **Certified absence in a complete audited trace:** every
   runtime-reachable, reference-aligned accepted proposal is a stable target
   accept, with no accepted-indeterminate, invalid, missing, unaudited, or
   unresolved-alignment event before the trace's reporting horizon.
3. **Universal kernel correctness:** every legal input in a declared domain
   refines the target operator. Prompt testing cannot establish this without a
   separate exhaustive or formal proof.

Claim 2 must additionally say either **conditional on this exact captured start
state** or **from this certified reference prefix**. A same-state conditional
audit must not be worded as an unconditional trajectory proof.

For non-random prompt collections, report exact counts and denominators rather
than a population confidence claim. Accepted-indeterminate events are an
explicit worst-case upper bound on additional false accepts for a fixed set of
otherwise complete, valid, reference-aligned gates. Later gates after alignment
becomes unresolved belong to a separate stratum.

## Approved Analytic Reference Contract

This section specifies the approved reference design for the initial profile.
It is not executable by itself; the eventual harness must implement and
validate every stated obligation rather than silently replacing one with a
backend comparison.

### Scope of the First Reference

The first tractable claim is a **conditional, per-trace KDA operator and state
refinement claim**. The audited operator:

1. begins at the stored outputs of the KDA layer's q/k/v, forget-gate, beta, and
   output-norm-gate projections, plus its persistent convolution and recurrent
   state;
2. includes causal convolution, SiLU, q/k normalization, safe decay, beta
   activation, the recurrent update, and gated RMSNorm; and
3. ends at the BF16 value consumed by `o_proj` and at every BF16/FP32
   persistent-state value that can become visible at a commit frontier.

The q/k/v and gate projections, `o_proj`, residual path, later KDA layers, MLA,
MoE, collectives, final norm, LM head, and score adjustment are outside this
local operator. They still matter to a token decision.

### Approved Surrounding-Model Contract

The first decision-level claim uses an **operational, bit-level hybrid target**:
replace each KDA operator, in model order, with the analytic operator projected
at its declared interfaces, while defining every non-KDA operation by one
frozen runtime executable. For a given injected BF16/FP32 interface state, the
exact FP32 score bits returned by that executable over the valid vocabulary are
the hybrid target scores. Pairwise differences are then exact differences of
those represented FP32 values; the surrounding runtime is not silently
reinterpreted as real-arithmetic linear algebra.

Making that operational map reproducible requires a versioned execution
manifest, including at least:

- source commit, container/build and package versions, CUDA/driver/JIT or cubin
  artifacts;
- GPU architecture, TP/DP world sizes and topology;
- kernel and collective algorithm choices, CUDA-graph/eager mode, environment
  variables, and all server arguments;
- exact model weights, request inputs, persistent state, and RNG state where
  applicable; and
- the exact KDA injection boundaries, tensor layouts, and interface bits.

This is an explicit implementation-relative assumption, not an absolute proof
of the real Kimi-K3 model. Independently checking each KDA layer on inputs
captured from the candidate run is insufficient after the first boundary
mismatch, because later captured inputs then have candidate rather than
reference provenance. If the manifested surrounding map is not reproducible
and demonstrably deterministic, the decision-level hybrid claim is
invalid/unclassified even though a local KDA same-state claim may still be
possible.

An absolute real-model argmax proof would additionally cover every projection,
MLA operation, MoE route and expert calculation, quantized operation,
collective, residual, norm, LM-head operation, and score adjustment. That
stronger claim is not the first milestone. Reports must say “KDA refinement
under the frozen surrounding runtime,” not simply “the target model is proved
correct.”

For the pinned model profile, the trace must record and verify:

- BF16 model activations;
- 93 transformer layers, of which 69 are KDA layers;
- 96 global KDA heads and 12 local heads under TP8;
- key and value head dimensions of 128;
- bias-free width-four q/k/v convolutions;
- safe-gate lower bound \(-5\);
- q/k normalization epsilon \(10^{-6}\);
- gated-RMSNorm epsilon \(10^{-5}\);
- BF16 convolution state; and
- FP32 recurrent state for the initial ReplaySSM audit profile.

These are profile facts, not universal KDA assumptions.

### Approved Initial Direct-Kernel Profile

The first direct-kernel audit profile is frozen to concurrency one, dense
block-size-8 DFlash verification, BF16 activation interfaces, BF16 convolution
state, FP32 persistent recurrent state, `APPLY_ONORM=true`, and
`CACHE_RING=true`. The audit must prove that `nv_cutedsl` actually dispatched
the direct CuTe kernel. It must separately certify the fused gated-RMSNorm
output, convolution state, live recurrent state, ReplaySSM ring
materialization, and every possible commit frontier.

Padded/ragged execution, `APPLY_ONORM=false`, `CACHE_RING=false`, different
state dtypes, and other candidate widths are separate profiles and inherit no
correctness conclusion merely from this first profile.

Selecting `nv_cutedsl` does not prove that the direct CuTe kernel ran. In the
current dispatcher, `nv_cutedsl` installs Triton as the fallback and calls the
direct kernel only when all of the following hold:

- CUDA and the CuTe DSL package are available on a device whose capability
  major is exactly 10;
- verification is dense and linear, with no ragged layout or tree-parent
  retrieval;
- total candidate width \(D\) is in \([2,8]\);
- the convolution is bias-free and the safe-gate lower bound is present;
- q, k, and v dimensions are all 128 with equal head counts;
- q/k/v, forget-gate, and beta entry tensors are BF16;
- convolution weights are FP32 with the required width-four layout;
- recurrent state is FP32 with the required contiguous/aligned layout;
- convolution histories have length three; and
- either full per-step recurrent snapshots have enough capacity or all
  four ReplaySSM rings have the wrapper-required shapes/dtypes and capacity
  \(L\geq D\): raw v/k are BF16 and log-decay/beta are FP32. The pool-created
  first profile additionally requires power-of-two \(L\geq2D\).

For DFlash, block size 8 means \(D=8\): one seed and seven proposals. It can
exercise the direct kernel. Block size 16 is outside the direct kernel's domain
and currently exercises the Triton fallback. After direct block-size-8
correctness is established, direct CuTe block-size-16 support is an explicit
follow-on milestone. It must refine the same frozen analytic operator and state
contract; extending the width must not redefine the oracle or relax the
certification criteria. Until that implementation exists and is certified,
block-size-16 measurements must be labeled as fallback results rather than
CuTe results.

A valid audit record must nevertheless prove the actual dispatch and
compile-time modes (`APPLY_ONORM` and `CACHE_RING`), not infer them from the
command line. It must also assert that every non-padding request has query
length exactly \(D\). The wrapper checks only aggregate fixed width.
CUDA-graph padding state slots are \(-1\), which skips state/ring access, but
repeated padding offsets can alias one zeroed dummy span and leave other
physical graph-tail entries uninitialized. Treat the entire graph tail as
logically invalid/uncovered and exclude it from numerical denominators.

The dispatch predicate itself probes only `replayssm_rawv` when full-state
scratch is absent. The called wrapper validates all four rings and raises on a
malformed contract; there is no exception-to-Triton fallback around the direct
call. A launch failure is invalid/unclassified, not evidence that Triton ran.

### Exact Inputs

For a same-state, layer-local reference evaluation, every captured tensor is
interpreted as its exact stored binary value:

- BF16: raw q/k/v projection values, raw forget-gate and beta logits,
  output-norm gate, and persistent convolution history;
- FP32: convolution weights, `A_log`, `dt_bias`, persistent recurrent state,
  the exactly promoted output-norm weight used by the fused path, and any other
  stored FP32 state;
- integers: request/state and scratch indices, sequence boundaries, token-row
  mappings, tensor sizes, strides, and layout metadata; and
- versioned constants: head dimension, convolution width, safe-gate lower
  bound, normalization epsilons, and query scale.

Token IDs, positions, tokenizer state, and request metadata establish trace
provenance and the row-to-context mapping. They are not BF16 numerical inputs
to the KDA operator. Backend scratch and ReplaySSM ring records are candidate
outputs or internal persistence mechanisms, not initial reference inputs.
The fused path's FP32 output-norm weight must be verified as an exact
value-preserving promotion of the pinned model parameter, not treated as an
independently chosen weight.

#### Approved Numeric Domain and Zero Projection

Decision: the analytic KDA operator has a **finite-only numeric domain**.

- A nonfinite value already present in an operator-entry tensor or constant
  makes the local conditional trace invalid/unclassified because its upstream
  state is outside this reference domain.
- A candidate kernel that produces NaN or infinity from valid finite inputs has
  a hard kernel failure; this is not numerical indeterminacy.
- A nonfinite eligible final score makes the decision trace invalid and is a
  hard audit failure. It must not be sanitized or normalized away.

Raw input bits are retained for provenance, but both input zero signs denote
the mathematical real value zero. At an explicit IEEE interface projection,
exact real zero canonically maps to \(+0\), and the resulting zero sign is
bit-significant for interface equality. Treating \(+0\) and \(-0\) as
interchangeable would require a separate proof that every downstream consumer
is sign-insensitive.

There is no uncertainty in a stored BF16 or FP32 input merely because it has low
precision: its represented dyadic value is exact input to this conditional
operator. Uncertainty enters through the definition/evaluation of the real
equations and, for trajectory-level claims, through the provenance of the
captured state.

The reference must record model revision, weight hashes, tokenizer revision,
target runtime configuration, tensor dtypes/shapes/strides, and the source of
every constant.

Decision: use **canonical IEEE FP32 constants**. Round each source/configuration
expression once with round-to-nearest, ties-to-even, then treat the resulting
FP32 bit pattern as an exact dyadic real throughout the analytic equations:

| Constant | Source expression | Canonical FP32 bits | Exact hexadecimal value |
| --- | --- | --- | --- |
| \(\epsilon_q=\epsilon_k\) | \(10^{-6}\) | `0x358637bd` | `0x1.0c6f7ap-20` |
| \(\epsilon_o\) | \(10^{-5}\) | `0x3727c5ac` | `0x1.4f8b58p-17` |
| \(L\) | \(-5\) | `0xc0a00000` | `-0x1.4p+2` |
| query scale | \(1/\sqrt{128}\) | `0x3db504f3` | `0x1.6a09e6p-4` |

These bits are part of the frozen model contract; they are not rediscovered
from Triton, CuTe, a compiler, or a disassembly. A candidate may materialize or
strength-reduce them however it likes, but its behavior is certified against
these exact values. A future model profile may deliberately choose different
constant precision only through an explicit semantic-contract change.

### Proposed KDA Equations

For each request, token \(t\), head \(h\), key channel \(k\), and value channel
\(v\), let \(x^q_{t,h,k}\), \(x^k_{t,h,k}\), and \(x^v_{t,h,v}\)
be raw projection values. Evaluate the real-valued KDA equations over the exact
represented input and weight values, with
\(\sigma(x)=1/(1+\exp(-x))\):

1. **Causal width-four convolution and SiLU**

   \[
   c^x_{t,h,c} =
       \sum_{r=0}^{3} w^x_{h,c,r} x^x_{t-3+r,h,c},
   \qquad
   u^x_{t,h,c} = c^x_{t,h,c}\,\sigma(c^x_{t,h,c}),
   \]

   for \(x \in \{q,k,v\}\), with \(c=k\) for q/k and \(c=v\) for v.
   The four taps are ordered oldest to current. Persistent history supplies
   the three positions before the invocation.

2. **Q/K normalization**

   \[
   q_{t,h,k} =
       \frac{u^q_{t,h,k}}
            {\sqrt{\sum_{\ell=0}^{127}(u^q_{t,h,\ell})^2 + \epsilon_q}}
       \frac{1}{\sqrt{128}},
   \qquad
   \kappa_{t,h,k} =
       \frac{u^k_{t,h,k}}
            {\sqrt{\sum_{\ell=0}^{127}(u^k_{t,h,\ell})^2 + \epsilon_k}}.
   \]

3. **Update strength and safe decay**

   \[
   \beta_{t,h} = \sigma(b_{t,h}),
   \]

   \[
   \lambda_{t,h,k} =
       L\,\sigma\left(
         \exp(A_{\log,h})\,(g_{t,h,k} + d_{h,k})
       \right),
   \qquad
   \alpha_{t,h,k} = \exp(\lambda_{t,h,k}),
   \]

   where \(L\) is the safe-gate lower bound and \(d\) is `dt_bias`.

4. **KDA recurrence**

   \[
   \widetilde H_{t,h,v,k} =
       \alpha_{t,h,k} H_{t-1,h,v,k},
   \]

   \[
   r_{t,h,v} =
       u^v_{t,h,v}
       - \sum_{k=0}^{127}
         \widetilde H_{t,h,v,k}\,\kappa_{t,h,k},
   \]

   \[
   H^{real}_{t,h,v,k} =
       \widetilde H_{t,h,v,k}
       + \beta_{t,h} r_{t,h,v}\kappa_{t,h,k},
   \qquad
   o_{t,h,v} =
       \sum_{k=0}^{127}H^{real}_{t,h,v,k}q_{t,h,k}.
   \]

5. **Fused gated RMSNorm and output**

   \[
   y^{real}_{t,h,v} =
       \frac{o_{t,h,v}}
            {\sqrt{\frac{1}{128}
              \sum_{\ell=0}^{127}o_{t,h,\ell}^2+\epsilon_o}}
       \omega_v\,\sigma(\mathrm{gate}_{t,h,v}),
   \]

where \(\omega_v\) is the shared per-value-channel gated-RMSNorm weight.

The initial audit operator includes this gated RMSNorm and therefore requires
the actual direct kernel to have `APPLY_ONORM=true`. If `APPLY_ONORM=false`,
the kernel boundary and the external fallback norm form a different composed
operator and require a separately stated contract; silently comparing the
pre-norm BF16 fallback boundary with the fused path would reintroduce a backend
rounding boundary into the oracle.

### Approved State-Composition Contract

Decision: use **stepwise FP32 persistent-state recurrence**.

After evaluating token \(t\)'s real-valued recurrence and current output,
project the persistent state once:

\[
H_t=\operatorname{RN}_{fp32}(H^{real}_t),
\]

and feed \(H_t\) into token \(t+1\). Thus \(H_{t-1}\) in the recurrence above
always denotes the exact real value represented by the prior token's canonical
FP32 state bits. The current token's \(o_t\) is computed from
\(H^{real}_t\) before that persistence projection.

This projection applies after every logical token, including positions inside
one verify block. It does not require a physical state store or synchronization
between token steps; a candidate may keep state resident in FP32 registers or
shared memory. It is a semantic persistence boundary.

Every verify position is a possible commit frontier, ordinary one-token decode
persists FP32 state at every token, and stepwise projection makes the reference
independent of whether the same token sequence is partitioned into sequential
decode, block size 8, or future block size 16. This is not Triton's BF16
recurrence-to-RMSNorm boundary, which remains excluded. The alternative
unrounded-across-the-block recurrence is not part of the initial contract.

### Semantic Interfaces Versus Backend Rounding

The analytic reference does not preserve an intermediate rounding boundary
merely because Triton, CuTe, FlashInfer, or ReplaySSM materializes a tensor
there. In particular, none of the following defines the oracle:

- Triton's BF16 recurrence output before normalization;
- a fallback backend's BF16 post-convolution q/k/v values;
- CuTe's FP32 accumulators, reduction order, FMA choices, fast exponentials,
  approximate reciprocal, or approximate reciprocal square root; or
- ReplaySSM's internal BF16 raw k/v and FP32 log-decay/beta representation.

The reference starts from exact stored operator-entry values and defines
real-valued internal variables. Backend casts and approximations are candidate
implementation errors to enclose and audit, not part of mathematical truth.
This is what keeps the reference useful while the CuTe kernel changes
aggressively.

The deployed numerical contract still has externally observable interfaces.
Under the approved first profile, project a certified real interval into:

- \(\operatorname{RN}_{bf16}(y^{real}_{t,h,v})\), the value consumed by
  `o_proj`;
- the FP32 recurrent state at every possible commit step;
- the BF16 raw projection values retained in each persistent width-three
  convolution window; and
- any additional value that leaves the audited operator and is consumed by a
  later invocation.

Here `RN_bf16` and `RN_fp32` mean IEEE round-to-nearest, ties-to-even only for
such an explicit interface projection. If a reference interval lies wholly in
one rounding cell and the candidate produces those bits, that interface value
is certified. If the bits differ, interface refinement has failed even when the
current argmax survives. A future, separately specified downstream proof may
show an output mismatch harmless for a particular decision, but the approved
initial audit does not attempt that propagation. A persistent-state mismatch
cannot be cleared by current-logit agreement: it is a state-transition failure
at that frontier. Exact rejoining of the complete target state at the same
certified emitted prefix can restore reference alignment at a later frontier.
Sound propagation can certify later token decisions over a bounded horizon,
but neither result retroactively clears the original state-transition failure.

### Approved Persistent-State Horizon

The initial audit certifies every possible commit frontier. A mismatch at a
counterfactual frontier is a definite state-refinement failure for that
frontier, but it does not end the observed runtime trajectory unless that
frontier is committed.

If the actually committed frontier has either a unique reference-state bit
pattern that differs from the candidate or an unresolved state projection:

- already-certified output and token decisions in the current block remain
  certified;
- reference alignment ends immediately after that commit and before the next
  target invocation;
- all subsequent proposal decisions are reported in the unresolved stratum;
  and
- the primary audit performs no bounded-horizon continuation and does not
  search for rejoining.

A future, separately specified diagnostic may restore target reference
alignment only by proving exact bitwise equality of the complete persistent
target state at the same certified emitted prefix. Equal tokens, logits, or one
KDA layer's state are insufficient. Resuming the exact speculative proposal
and performance trajectory additionally requires the relevant draft state to
be restored; draft-state equality is not required merely to restore target
semantics.

ReplaySSM is an implementation of persistence and reconstruction, not the
semantics of KDA. Its eventual committed state and persistent history must
refine the real recurrence and its declared external storage projections; its
internal representation cannot be used to define the reference.

### ReplaySSM Refinement Obligation

ReplaySSM creates a second numerical path that must be audited independently of
the live verify output:

- the direct CuTe kernel computes its live recurrence from FP32 post-SiLU q/k/v
  values and its own q/k normalization and transcendental approximations;
- in `CACHE_RING` mode it stores post-SiLU, pre-normalization k and post-SiLU v
  to BF16 rings, while storing log decay and activated beta in FP32 rings; and
- at commit, a Triton fold reloads those ring values, renormalizes k, and
  replays the accepted prefix into the FP32 checkpoint with its own operation
  order and transcendental implementation.

Consequently, the committed recurrent state is not guaranteed to be the state
that produced the accepted CuTe outputs. The code's “bitwise clone” claim is
relative to a particular Triton recurrent baseline given the rounded ring
inputs; it is not an analytic certificate and not a clone of the live CuTe
trajectory.

For a block of width \(D\), the audit must certify the externally committed
recurrent and convolution state for every legal commit length
\(c\in[1,D]\), where \(c=k+1\) consumes
`[seed, p_1, ..., p_k]`. The ReplaySSM rings, full-state scratch, and
intermediate convolution buffers may be captured for diagnosis, but they are
internal evidence. The semantic outputs are the selected persistent
checkpoint and convolution history. Target state and draft KV must be reported
as separate commit obligations.

### Reference Arithmetic and Enclosure

Ordinary FP64 is a useful fast path but is not the normative proof. The
normative reference should:

- decode every BF16/FP32 value exactly;
- give every primitive, including each reduction, an outward enclosure;
- evaluate algebraic operations with exact rational accumulators where
  practical, or use a documented outward-rounded accumulation scheme;
- evaluate `exp`, `sqrt`, sigmoid, and related functions using
  outward-directed MPFR/Arb intervals;
- use center-radius or affine forms where plain interval boxes lose important
  cancellation;
- increase working precision until the enclosure proves a unique interface
  rounding result or a tie-aware decision;
- terminate only with a proof or a reason-coded resource/precision limit; and
- handle midpoint parity, subnormals, signed zero, overflow boundaries, and
  exact ties according to the declared IEEE projection and token-order rules.

Formally, a unique bit certificate proves that the real enclosure is contained
in the full preimage \(\operatorname{RN}^{-1}(b)\) of one BF16/FP32 bit pattern.
Touching a rounding boundary requires an exact comparison and the destination
significand's parity; enclosure width alone is insufficient.

The reference should expose both a high-precision center and a certified lower
and upper endpoint for every observable result. At least one independently
implemented evaluator or a set of exact/symbolic special cases should guard
against correlated bugs in the primary interval implementation. A precision
ladder, nested intervals, an algebraically independent recurrence formulation,
and invariants such as

\[
\exp(L) \leq \alpha \leq 1,\quad
0 \leq \beta \leq 1,\quad
\lVert k\rVert_2 \leq 1,\quad
\lVert q\rVert_2 \leq 1/\sqrt{128}
\]

should validate the reference without treating an existing GPU kernel as its
oracle. The decay invariant assumes \(L\leq0\), which is true for the pinned
profile's \(L=-5\); it is not a universal KDA invariant. These checks are
necessary sanity tests, not proof that every equation or index is correct.

### Conditional State and Trajectory State

Two proof levels must remain separate:

1. **Same-state conditional proof:** treat the captured persistent recurrent
   and convolution states as exact inputs and certify this verify invocation.
   This isolates the candidate kernel. It conditions on those values; it does
   not certify their provenance or endorse the backend that produced them.
2. **Reference-trajectory proof:** reconstruct or inductively certify the
   recurrent and convolution states from an earlier certified prefix. This is
   required before claiming that the entire generated trajectory follows the
   analytic target contract. A captured backend state cannot substitute for
   this proof.

The induction must follow the ordered model graph, not audit 69 KDA layers
independently on candidate-derived inputs. If one KDA invocation's output and
state project to the certified interface bits, the frozen surrounding
operations receive identical inputs and can carry the induction to the next
KDA invocation. At the first mismatching boundary, later captured layer inputs
lose reference provenance. Continuing would require a hybrid replay from that
boundary or a sound enclosure through the remaining graph; the approved
initial audit instead stops that sample at the first mismatch.

A reference trajectory also needs a certified starting point. PTX prefill,
Triton prefill, non-spec decode, and a saved runtime checkpoint are evidence,
not automatically valid initial analytic state.

The approved staging is:

1. The first kernel milestone is a **same-state conditional audit** using the
   exact captured block-entry state bits. Its local KDA refinement and
   decision-level conclusions must be labeled “conditional on the exact
   captured start state”; neither a large corpus nor zero observed failures
   certifies the provenance of that state.
2. An **end-to-end generation-correctness claim** is prohibited until the
   prompt prefix has been reconstructed from specified empty model state
   through the analytic-KDA/frozen-runtime hybrid, or reached inductively from
   a complete checkpoint whose prefix and target state were already certified.

### Profiled and Synthetic Conditional Inputs

Fast kernel iteration may use a versioned offline input corpus without
reloading the full model. That corpus should contain:

- exact-bit captures of real KDA boundary tensors and block-entry states, with
  layer/head/token/position, shape, stride, layout, and dispatch metadata;
- mutation fuzzing around captures, including representable one-ULP neighbors
  and values near rounding-cell boundaries;
- structured finite stress cases for q/k normalization near epsilon, the
  safe-gate clamp, sigmoid/decay extremes, recurrence cancellation,
  gated-RMSNorm near epsilon, BF16 projection boundaries, signed zero, and
  large-but-finite magnitudes; and
- broad seeded random finite inputs satisfying the declared dtype, shape,
  layout, head mapping, and operator-domain constraints.

Observed production distributions guide sampling density but do not narrow the
finite input contract. Synthetic cases may be valid operator inputs without
being reachable model states; their results are therefore local same-state
claims, not trajectory error rates. A failing case is a valid conditional
counterexample. Passing any finite fuzz corpus, regardless of size, is not a
universal proof and does not replace certified prompt trajectories or the
sound interval/rounding argument.

### Certification Cascade

The eventual design should use the following cascade:

1. Reject structural failures, nondeterminism, invalid indexing, races,
   nonfinite eligible scores, malformed row mappings, missing data, or
   out-of-bounds writes as invalid/unclassified before numerical analysis.
   Record direct-versus-fallback dispatch, `APPLY_ONORM`, `CACHE_RING`, valid
   query lengths, graph-padding rows, and state/scratch maps.
2. Evaluate the analytic KDA reference for the exact trace.
3. Certify output, convolution state, and recurrent state separately at every
   possible commit frontier. If every interval rounds uniquely to the same
   external bits, the candidate refines this operator invocation. Identical
   output bits allow induction through the frozen surrounding computation;
   identical state bits preserve the next-block induction.
4. If a unique reference output bit pattern differs, record an
   interface-refinement failure, preserve the complete counterexample, and stop
   that sample's reference trajectory at the first mismatch. If the enclosure
   spans rounding cells, first tighten it or mark bit refinement indeterminate
   and stop under the same rule. Downstream accepted proposals become
   accepted-indeterminate and downstream rejected proposals become
   rejected-indeterminate; neither is relabeled as a false accept or false
   reject without a sound downstream proof. The initial audit performs no
   hybrid replay. A future exact hybrid replay or sound enclosure may be used
   as an optional diagnostic under a separately frozen contract, but an
   ordinary approximate GPU “shadow” forward is not such a proof.
5. If a unique persistent-state bit pattern differs, record a state-transition
   failure; if the state enclosure does not round uniquely, record unresolved
   state certification. Do not clear either with current-logit agreement. A
   counterfactual-frontier failure is recorded without ending the observed
   trajectory. At an actually committed failing or unresolved frontier,
   preserve already-certified current-block decisions, end reference alignment
   before the next target invocation, and classify all subsequent gates as
   unresolved. The initial audit performs no bounded continuation or rejoin
   search.
6. Use coarse all-vocabulary bounds to eliminate clear competitors and
   expensive exact/outward dot products only for surviving tokens.
7. Apply the tie-aware pairwise tests to proposals and independently certify
   every emitted replacement/bonus winner. Classify proof exhaustion as
   indeterminate with a reason, not as a tolerance-based tie.

For the initial operational hybrid, the normative decision values are the
manifested runtime's returned FP32 score bits, and the pairwise certificate
subtracts those exact represented values. If a later fully analytic surrounding
contract is adopted—or a sound proof shows equivalence to the operational
map—an affine LM head can instead use direct analytic differences,

\[
z_p-z_j=(w_p-w_j)^T h+(b_p-b_j),
\]

because they retain cancellation and are tighter than subtracting two
independently bounded analytic logits. This formula does **not** by itself equal
a rounded GEMM, quantized LM head, collective, or FP32 conversion and therefore
cannot replace the operational score map without that proof. Soft-capping,
custom score adjustments, masking, grammar, or a different LM-head
implementation must also be modeled in their actual order. The initial strict
profile excludes request-level adjustments and grammar, but its execution
manifest still covers valid-vocabulary slicing, relevant TP/DP
gather/collective behavior, FP32 score conversion, and the first-index argmax
rule.

### Initial Contract Decision Status

No unresolved definition or design decision blocks implementation of the
approved initial profile. Expanding the claim to another candidate width,
layout, dtype, dispatch mode, start-state provenance level, downstream replay,
or post-mismatch horizon requires a separately versioned profile.

## Reporting Principles

Every future report must identify its exact unit and denominator: request,
block, proposal gate, emitted replacement/bonus, KDA invocation, KDA interface
element, or possible commit frontier. It must record:

- repository/model revisions and the full frozen target configuration;
- selector names **and actual** direct/fallback dispatch, candidate width,
  `APPLY_ONORM`, `CACHE_RING`, dtypes, and state mode;
- the row's exact \((P,s,p_1,\ldots)\) context and whether it is
  runtime-reachable, reference-aligned, unresolved, post-divergence, or
  counterfactual;
- correct accepts, correct rejects, false accepts, false rejects,
  accepted-indeterminate, rejected-indeterminate, and invalid/unclassified
  proposal decisions, each with explicit denominators;
- independently certified replacement/bonus token identities;
- KDA output-interface, recurrent-state, convolution-state, and draft-KV
  results as separate axes;
- first certified failure/unresolved event per trajectory, without counting its
  causal descendants as independent pre-divergence errors;
- verifier-committed, raw emitted, and user-visible token runs separately; and
- a reason code and achieved precision/bound width for every indeterminate or
  invalid event.

When indeterminate cases remain, report certified counts and explicit
worst-case ranges rather than an exact error rate. On a fixed set of valid,
reference-aligned gates, observed false accepts are the lower bound and adding
accepted-indeterminate gates gives the conservative upper bound. Invalid,
missing, or unaudited accepted gates must also be counted as possible failures
or else no finite reported upper bound is justified. Gates after alignment
becomes unresolved are not silently added to the aligned denominator; they are
reported in the unresolved/post-divergence stratum.

The phrase **certified absence of false accepts in the audited trace** is
reserved for a complete trace in which every runtime-reachable,
reference-aligned accepted proposal before the reporting horizon is a certified
stable accept, with no accepted-indeterminate, invalid, missing, unaudited, or
unresolved-alignment event. It must be qualified either **conditional on the
exact captured start state** or **from a certified reference prefix**. It is not
a universal kernel-correctness claim and does not follow from an observed count
of zero.

## Deferred Follow-on Profiles

The initial contract does not cover direct CuTe block size 16, padded/ragged
execution, alternative state dtypes or compile-time modes, downstream hybrid
replay, or automatic rejoin detection. Prompt reconstruction remains required
before promoting same-state conditional results to end-to-end
generation-correctness claims. Each extension must preserve the analytic KDA
semantics and explicitly freeze its additional operational assumptions.

## Change Log

### 2026-08-02

- Started a clean, first-principles correctness investigation.
- Recorded provisional greedy acceptance terminology.
- Separated stable, indeterminate, reachable, and counterfactual outcomes.
- Separated numerical, decision, emitted-token, and end-to-end correctness.
- Recorded that non-speculative and speculative target execution can diverge
  even with zero false accepts.
- Distinguished divergent and non-divergent false rejects and listed other
  zero-false-accept divergence mechanisms.
- Classified false accepts as correctness failures and false rejects as
  performance-only only when token and state recovery remain correct.
- Defined accepted-indeterminate and rejected-indeterminate outcomes and
  prohibited silently resolving them in the draft's favor.
- Documented SGLang's shifted greedy DFlash layout: returned acceptance length
  counts both an emitted bonus/replacement and an equal-length but different
  run of committed verify inputs; the emitted endpoint is not yet in state.
- Distinguished performance-only accept-prefix truncation from a certified
  target-row false reject whose differing argmax is emitted immediately.
- Established that the backend-independent analytic reference and final-logit
  decision contract must be frozen before CuTe optimization resumes.
- Added a provisional MPFR/interval KDA reference contract and an
  interface-first certification cascade.
- Removed Triton, CuTe, FlashInfer, and ReplaySSM internal materialization
  boundaries from the mathematical KDA definition; they are implementation
  effects to audit, not reference semantics.
- Strictly reviewed the worklog against current DFlash scanning, target-state
  commit, KDA dispatch, direct CuTe, fused gated RMSNorm, and ReplaySSM code.
- Replaced scalar-margin language with first-index tie-aware pairwise
  certification and separated invalid traces from numerical indeterminacy.
- Split runtime reachability from certified reference alignment and added
  output, recurrent-state, convolution-state, and draft-KV correctness axes.
- Corrected the KDA operator boundary, tensor dtypes, head-indexed equations,
  direct-kernel eligibility, CUDA-graph padding treatment, and block-size-16
  fallback semantics.
- Identified ReplaySSM commit folding as a separate numerical trajectory from
  the live CuTe output and required every possible commit frontier to be
  certified.
- Corrected CUDA-graph padding and ReplaySSM launch-contract details: physical
  graph tails are not all guaranteed zero, and malformed rings fail the direct
  launch rather than falling back.
- Recorded direct CuTe block-size-16 support as a follow-on milestone after
  block-size-8 correctness, governed by the same frozen reference and
  certification criteria.
- Defined the proposed surrounding-model claim as an operational bit-level
  hybrid with a complete execution manifest, rather than treating a
  deterministic GPU runtime as implicit real arithmetic.
- Added an explicit claim hierarchy: observed zero failures, certified absence
  in a complete trace, and universal kernel correctness.
- Recorded the blocking choice between unrounded block state and stepwise FP32
  persistent-state composition, then approved stepwise FP32 state so reference
  semantics remain invariant to speculative block partitioning.
- Approved canonical IEEE FP32 normalization, safe-gate, and query-scale
  constants and recorded their exact bit patterns.
- Approved a finite-only analytic domain, canonical \(+0\) projection for exact
  real zero, and bit-significant zero signs at external interfaces.
- Approved the operational bit-level hybrid as the first decision-level target,
  with a versioned execution manifest and invalid/unclassified decision traces
  whenever the frozen surrounding map is not reproducible and deterministic.
- Approved same-state conditional audits as the first kernel milestone while
  reserving end-to-end generation-correctness claims for analytically
  reconstructed or inductively certified prefixes.
- Added exact captured-input replay, capture-guided mutation, structured
  numerical stress cases, and seeded finite random fuzzing as offline
  conditional tests, without treating sampled success as universal proof.
- Froze the first direct-kernel profile to dense block-size-8, concurrency-one
  execution with BF16 interfaces, FP32 recurrent state, fused gated RMSNorm,
  and ReplaySSM ring caching; all other modes remain separate profiles.
- Approved stopping each initial-audit sample at its first KDA output mismatch
  or unresolved interface projection, preserving the counterexample and
  conservatively classifying downstream decisions as accepted- or
  rejected-indeterminate instead of performing hybrid replay.
- Approved ending reference alignment before the next target invocation after
  an actually committed state mismatch or unresolved state projection, while
  preserving already-certified current-block decisions and recording
  counterfactual-frontier state failures separately.
- Closed all blocking definition and design choices for the initial
  block-size-8 direct-kernel correctness profile.
- Explicitly deferred all harness construction, testing, and optimization.
