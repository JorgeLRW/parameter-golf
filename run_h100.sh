#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# RunPod 8×H100 — Route110 + Shift50 (full competition config)
# ═══════════════════════════════════════════════════════════════════
#
# Usage:
#   1. SSH into your RunPod instance
#   2. cd /workspace && git clone <your-repo> parameter-golf && cd parameter-golf
#   3. python3 data/cached_challenge_fineweb.py --variant sp1024
#   4. cp train_gpt_experimental.py train_gpt.py
#   5. chmod +x run_h100.sh && ./run_h100.sh
#
# Trains 600s on 8×H100, then full Hessian GPTQ + sliding window eval.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Architecture ──
export MLP_ACT="leakyrelu2"
export MLP_RESID_GATE="1"
export MRG_GATE_SLOPE="4.0"
export ROUTE_ATTN_LAYERS="1,2,3,4,5,6,7,8,9,10"
export TOKEN_SHIFT_FRAC="0.5"
export BIGRAM_VOCAB_SIZE="3072"
export BIGRAM_DIM="112"
export LOGIT_SOFTCAP="30.0"

# ── Attention ──
export ATTN_FREEZE_STEP="5"
export ATTN_DETACH_INPUT_AFTER_FREEZE="1"
export ATTN_SDP_BACKEND="flash"

# ── NCCL / RunPod compatibility ──
export NCCL_CUMEM_ENABLE="0"
export NCCL_NVLS_ENABLE="0"

# ── Training ──
# TRAIN_BATCH_TOKENS=786432 (default, correct for 8×H100)
# TRAIN_SEQ_LEN=2048 (default)
export ITERATIONS="20000"
# Small safety buffer below the hard 600s challenge cap.
export MAX_WALLCLOCK_SECONDS="599.5"
export WARMUP_STEPS="0"
export WARMDOWN_ITERS="4000"
export USE_TORCH_COMPILE="1"
export RUN_ID="${RUN_ID:-h100_route110_shift50}"
export SEED="${SEED:-1337}"
export TARGET_BYTES="15900000"

# ── GPTQ (AR self-gen calibration, matching #1 entry) ──
export GPTQ_CALIB_SOURCE="ar"
export GPTQ_CALIB_NUM_SEQS="64"
export GPTQ_CALIB_TEMPERATURE="0.8"

# ── Logging ──
export VAL_LOSS_EVERY="200"
export TRAIN_LOG_EVERY="50"

echo "═══════════════════════════════════════════════════"
echo "  Route110 + Shift50 — 8×H100 — 600s"
echo "  BigramHash 3072×112, warmdown 4000"
echo "  GPTQ: AR self-gen calib, LZMA preset=9"
echo "  Run ID: ${RUN_ID}  Seed: ${SEED}"
echo "═══════════════════════════════════════════════════"

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Done! Check final_int6 lines above for BPB."
echo "═══════════════════════════════════════════════════"
