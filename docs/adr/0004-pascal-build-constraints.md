# ADR 0004: Pascal (sm_61) build constraints — measured traps on the GTX 1070

- Status: accepted
- Date: 2026-10-02
- Source: measured on this machine (i7-6700K, GTX 1070 8GB, Arch), preserved from
  beads robot-baa/a3f/gsg/yqu/ut2/49t/b4w and commits 28e0a33, 9f6e673, 65e043d

## Decision

All CUDA artifacts on this machine are built or pinned with sm_61 explicitly in mind.
No random third-party prebuilt wheels. Every binary/model records its revision,
build flags, and measured fit (see docs/pins.yaml).

## Measured constraints (treat as physics, re-verify on any bump)

- **torch wheels**: cu128 dropped sm_61; CUDA 13 dropped Pascal entirely. The default
  cu130 wheel reported "cuda available: True" then could not launch a single kernel
  (arch_list sm_75+). **cu126 is the last wheel line containing sm_61**; torch and
  torchaudio must be a matched pair (a mismatch breaks torchaudio's abi3 .so and takes
  Silero VAD down with it). Verify `torch.cuda.get_arch_list()` contains `sm_61`
  after any bump.
- **kokoro-onnx with the CUDA EP does NOT work here** (cuDNN RNN init error). The GPU
  TTS path is torch, not ONNX Runtime.
- **llama.cpp**: build with `-DCMAKE_CUDA_ARCHITECTURES=61` and
  `-DGGML_CUDA_F16=OFF` — GP104 runs fp16 at 1/64 of fp32 rate, so F16 math must stay
  off. FA and Q4 kernels work on sm_61, just without tensor cores. No bfloat16, no
  FlashAttention 2. MTP speculative decoding (~1.7x, draft-mtp) landed upstream
  2026-05-16 (PR #22673) but requires n_parallel=1 and a Pascal BF16 fallback patch
  of uncertain merge status — verify before relying on it.
- **mmproj dtype**: never load a BF16 vision projector on this card — Pascal lacks
  BF16, it costs ~675 MB of VRAM to emulate, and llama.cpp reports unimplemented CUDA
  ops for it. Use F16.
- **CPU quantization is a trap**: Skylake has AVX2 but no AVX-512/VNNI, so int8 GEMMs
  hit slow paths. Measured: kokoro-onnx int8 3.87 s vs fp32 0.86 s (5x REGRESSION);
  Parakeet int8 slower than fp32 for any clip over 2 s. **Stay fp32 everywhere on
  CPU.** CTranslate2 supports only int8/int8_float32 — one more reason STT runs ONNX.
- **VRAM budget** (measured, 8 GB card): llama.cpp Q4_K_M 8K ~4.4 GB (UD-Q4_K_XL is
  +0.68 GB for strictly better quality than Q4_K_M), Kokoro-onnx-GPU ~904 MiB +
  ~250 MB CUDA context. Whisper/Parakeet int8 ~1 GB. Peak for long replies is
  untested territory — measure before adding any second GPU-resident model.
- **reachy-mini SDK**: bumping deps must be targeted (`uv lock
  --upgrade-package X`) — a blanket upgrade pulls a newer torch whose wheels drop
  sm_61. 1.8.x was the sweet spot (audio resilience, no-180-deg-interpolation goto);
  >=1.9 changes daemon/BLE API and CORS.
