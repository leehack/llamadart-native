#pragma once

#include "ggml.h"

// Scheduler work between chunk boundaries inside one TTS step: multiply-adds
// for MUL_MAT, elements for other computed nodes.
static constexpr double llama_dart_tts_eval_budget = 2.5e9;

struct llama_dart_tts_eval_chunker {
  double budget = llama_dart_tts_eval_budget;
  double work = 0.0;
  bool boundary_due = false;
  bool past_first_boundary = false;
};

static inline double llama_dart_tts_eval_node_work(const ggml_tensor *node) {
  switch (node->op) {
  case GGML_OP_NONE:
  case GGML_OP_VIEW:
  case GGML_OP_RESHAPE:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
    return 0.0;
  case GGML_OP_MUL_MAT:
    return static_cast<double>(ggml_nelements(node)) *
           static_cast<double>(node->src[0]->ne[0]);
  default:
    return static_cast<double>(ggml_nelements(node));
  }
}

// Answers one ggml_backend_sched_eval_callback query for a step in progress.
//
// Budget boundaries fall only after a MUL_MAT node. A boundary splits any
// fusion that spans it, which can change the output; the CPU and Metal
// backends fuse nothing that continues past a MUL_MAT.
//
// A break stops only the current scheduler split; later splits still run.
// The code predictor runs first in a step and feeds its sampled indices to
// get_rows, so no break happens before the step's first budget boundary.
static inline bool llama_dart_tts_eval_answer(
    llama_dart_tts_eval_chunker *chunker, const ggml_tensor *node, bool ask,
    bool cancelled) {
  if (!ask) {
    return !cancelled;
  }
  if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
    return false;
  }
  if (cancelled && chunker->past_first_boundary) {
    return true;
  }
  chunker->work += llama_dart_tts_eval_node_work(node);
  if (chunker->work >= chunker->budget) {
    chunker->boundary_due = true;
  }
  if (!chunker->boundary_due || node->op != GGML_OP_MUL_MAT) {
    return false;
  }
  chunker->work = 0.0;
  chunker->boundary_due = false;
  chunker->past_first_boundary = true;
  return true;
}
