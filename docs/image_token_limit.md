# Opt-in image-token limit

The next wrapper rebuild adds `llama_dart_mtmd_supports_image_token_limit`.
It reads GGUF projector metadata without weights or GPU initialization and
returns 1 only for `gemma4v`/`gemma4uv` and limits 70 through 1120. Both the
legacy `clip.projector_type` and mixed-modality `clip.vision.projector_type`
keys follow upstream precedence. Missing, unknown and wrongly typed keys
return unsupported; read/argument errors return -1 through the exception
barrier. The call is guarded against exit teardown.

After a successful probe, callers explicitly set
`mtmd_context_params.image_max_tokens` before the existing guarded projector
initialization. Omitted options retain upstream metadata defaults. The option
reduces vision-patch detail and may change answers; it does not automatically
track a context microbatch or bound prompt/marker tokens. Existing decode
size checks remain necessary. Gemma 3 and other projector families are
unsupported rather than silently ignoring the request.

The 70-token lower bound preserves the default minimum-pixel threshold;
the 1120-token upper bound prevents expanding the model's default budget.
Primary implementation: upstream v0.6.0 clip.cpp Gemma 4 projector setup and
clip-model.h `set_limit_image_tokens`/`set_warmup_n_tokens`.

`llamadart_image_token_limit_test` reads real metadata fixtures and covers
supported ranges, unsupported types, mixed metadata, type mismatch and
exception-barrier recovery. Real Gemma 4 image encoding remains a separate
qualification gate before publication or downstream pin adoption.

Local verification (2026-10-09, macOS arm64, upstream v0.6.0): all 46 native
CTest cases pass and the export validator finds all 86 required exports.
A real cached Gemma 4 E2B Q4_K_S model with its F16 projector and a 1600x1600
RGB image produces 1089 image tokens by default and 484 with an explicit
512-token limit. The same maintained test optionally takes a model and
projector path and checks both preprocessing and vision encoding:

```sh
build/v060-2/llamadart_image_token_limit_test /path/to/model.gguf /path/to/mmproj.gguf
```

The encoding smoke is still running at the time of this candidate commit;
preprocessing results alone are not an encoding or answer-quality claim.
Real noncausal Gemma 4 (`gemma4v`) projectors remain unqualified: no matching
cached projector is available. Gemma 3 remains explicitly unsupported. No
runtime release, device run or downstream pin change was performed.
