# Sampler selection guard

Issue: [llamadart-native#238](https://github.com/leehack/llamadart-native/issues/238)
(replaces the closed [#235](https://github.com/leehack/llamadart-native/issues/235),
whose closing change [llamadart#2316](https://github.com/leehack/llamadart/pull/2316)
validates only Dart inputs). Consumer follow-up:
[llamadart#2320](https://github.com/leehack/llamadart/pull/2320) and
[llamadart#2295](https://github.com/leehack/llamadart/issues/2295).

## Problem

`llama_sampler_sample` applies the chain, selects a candidate and accepts it in
one call. When every candidate ends the chain at `-inf`, it can still select
one. A grammar sampler that accepts an end-of-generation token before its
grammar is finished reaches `GGML_ABORT` in `llama_grammar_accept_impl`
(`llama-grammar.cpp`), which no exception barrier catches: the process ends
with SIGABRT. A chain that selects nothing reaches the `GGML_ASSERT` on
`selected` in `llama_sampler_sample` the same way.

## Contract

`llama_dart_sampler_sample` and `llama_dart_sampler_sample_and_accept_n` apply
the chain themselves, as `llama_sampler_sample` does at llama.cpp `v1.0.0` and
`v1.1.0`, and check the selection before the sampler accepts it. They fail,
without accepting the token, when the chain selected no candidate or the
selected candidate's logit is not finite (`-inf`, `+inf` or NaN):

| Export | On a rejected selection |
| --- | --- |
| `llama_dart_sampler_sample` | `LLAMA_TOKEN_NULL` |
| `llama_dart_sampler_sample_and_accept_n` | `LLAMA_DART_STATUS_EXCEPTION`; the rejected token is neither accepted nor written to `out_tokens` |

Both also set `llama_dart_last_error` to `The sampler selected no candidate` or
`The sampler selected a candidate with a non-finite logit`. They are the same
failure value and last error as for a caught exception, so existing handling
applies: free the sampler, or `llama_sampler_reset` it and start over. The
batched export stops at the failing step. The tokens of earlier steps were
accepted by the sampler, and their count is not returned.

Unchanged: a token that the context's backend sampler chose is accepted once
and returned without this check, and a valid selection is accepted once and
returned as before. No export, signature or struct changes, so there is no ABI
delta and the Dart bindings do not change.

The internal chains of the TTS and live ASR sessions call `llama_sampler_sample`
directly. They hold no grammar sampler and are unchanged.

## Consumer handling

A binding that wants a typed error must treat a bare `LLAMA_TOKEN_NULL` from
`llama_dart_sampler_sample` as a failure, not only a caught exception: the last
error is set in both cases. `llamadart` does that after it pins an artifact that
contains this guard.

## Validation

`llamadart_exit_teardown_model-nonfinite-selection` covers both exports with an
all `-inf` chain under an unfinished grammar that selects the end of
generation, NaN and `+inf` selections, a chain that selects nothing, a failure
at the second step of a batch, and valid sampling afterwards. Without the check
the first case ends the process with SIGABRT.
