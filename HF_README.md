---
license: apache-2.0
pretty_name: FARBench Docker Images
tags:
  - benchmark
  - docker-image
  - research-agent
---

# FARBench Docker Images

Per-task Docker images, saved with `docker save | gzip` and uploaded as plain LFS files. Each tarball is a fully self-contained image (CUDA + Python + task deps + baked task data); load it with `docker load` and the resulting image carries the canonical tag `farbench/farbench:<task>-<cuda>`.

Files are at the repo root with the flat naming convention **`<task>-<cuda>.docker.tar.gz`** (e.g. `domainnet_quickdraw-cu118.docker.tar.gz`).

## Website

Explore the FARBench leaderboard, agent trajectories, and capability analysis at:

- https://anomyous4.github.io/anomyous-website/#/

## Quick start

```bash
# 1. Download the tarball you need (single task, cu118 example).
huggingface-cli download \
    FARBenchAnonymous/FARBench \
    domainnet_quickdraw-cu118.docker.tar.gz \
    --repo-type dataset \
    --local-dir ./farbench-images

# 2. Load into your Docker daemon.
docker load -i ./farbench-images/domainnet_quickdraw-cu118.docker.tar.gz

# 3. The image is now available locally as e.g.
#    farbench/farbench:domainnet_quickdraw-cu118
docker images | grep farbench
```

If a tarball is split (`*.part00`, `*.part01`, ...) merge first:

```bash
cat <task>-<cuda>.docker.tar.gz.part* > <task>-<cuda>.docker.tar.gz
docker load -i <task>-<cuda>.docker.tar.gz
```

## Available tasks

| Task | Domain | Metric | `cu118` | `cu128` |
|------|--------|--------|------|------|
| `ade20k` | computer vision | `mIoU` | [ade20k-cu118](./ade20k-cu118.docker.tar.gz) | [ade20k-cu128](./ade20k-cu128.docker.tar.gz) |
| `aime_math_rl` | natural language processing | `exact_match` | [aime_math_rl-cu118](./aime_math_rl-cu118.docker.tar.gz) | [aime_math_rl-cu128](./aime_math_rl-cu128.docker.tar.gz) |
| `assist2009_kt` | natural language processing | `auc` | [assist2009_kt-cu118](./assist2009_kt-cu118.docker.tar.gz) | [assist2009_kt-cu128](./assist2009_kt-cu128.docker.tar.gz) |
| `asvspoof2021_la` | audio/speech | `eer` | [asvspoof2021_la-cu118](./asvspoof2021_la-cu118.docker.tar.gz) | [asvspoof2021_la-cu128](./asvspoof2021_la-cu128.docker.tar.gz) |
| `bigcodebench_codegen` | natural language processing | `pass_at_1` | [bigcodebench_codegen-cu118](./bigcodebench_codegen-cu118.docker.tar.gz) | [bigcodebench_codegen-cu128](./bigcodebench_codegen-cu128.docker.tar.gz) |
| `cifar100lt` | computer vision | `balanced_accuracy` | [cifar100lt-cu118](./cifar100lt-cu118.docker.tar.gz) | [cifar100lt-cu128](./cifar100lt-cu128.docker.tar.gz) |
| `cifar100n` | computer vision | `accuracy` | [cifar100n-cu118](./cifar100n-cu118.docker.tar.gz) | [cifar100n-cu128](./cifar100n-cu128.docker.tar.gz) |
| `climsim_lowres` | AI for science | `mean_r2` | [climsim_lowres-cu118](./climsim_lowres-cu118.docker.tar.gz) | [climsim_lowres-cu128](./climsim_lowres-cu128.docker.tar.gz) |
| `cogniplan` | robotics | `plan_success` | [cogniplan-cu118](./cogniplan-cu118.docker.tar.gz) | [cogniplan-cu128](./cogniplan-cu128.docker.tar.gz) |
| `crohme_hmer` | computer vision | `exprate` | [crohme_hmer-cu118](./crohme_hmer-cu118.docker.tar.gz) | [crohme_hmer-cu128](./crohme_hmer-cu128.docker.tar.gz) |
| `domainnet_quickdraw` | computer vision | `accuracy` | [domainnet_quickdraw-cu118](./domainnet_quickdraw-cu118.docker.tar.gz) | [domainnet_quickdraw-cu128](./domainnet_quickdraw-cu128.docker.tar.gz) |
| `etth1_forecasting` | AI for science | `mse` | [etth1_forecasting-cu118](./etth1_forecasting-cu118.docker.tar.gz) | [etth1_forecasting-cu128](./etth1_forecasting-cu128.docker.tar.gz) |
| `flip_aav` | AI for science | `spearman` | [flip_aav-cu118](./flip_aav-cu118.docker.tar.gz) | [flip_aav-cu128](./flip_aav-cu128.docker.tar.gz) |
| `habitat3` | robotics | `nav_seek_success` | [habitat3-cu118](./habitat3-cu118.docker.tar.gz) | [habitat3-cu128](./habitat3-cu128.docker.tar.gz) |
| `humanoidbench` | robotics | `success_rate` | [humanoidbench-cu118](./humanoidbench-cu118.docker.tar.gz) | [humanoidbench-cu128](./humanoidbench-cu128.docker.tar.gz) |
| `iwildcam_wilds` | AI for science | `macro_f1` | [iwildcam_wilds-cu118](./iwildcam_wilds-cu118.docker.tar.gz) | [iwildcam_wilds-cu128](./iwildcam_wilds-cu128.docker.tar.gz) |
| `metrla_traffic` | AI for science | `mae` | [metrla_traffic-cu118](./metrla_traffic-cu118.docker.tar.gz) | [metrla_traffic-cu128](./metrla_traffic-cu128.docker.tar.gz) |
| `minigrid` | robotics | `return` | [minigrid-cu118](./minigrid-cu118.docker.tar.gz) | [minigrid-cu128](./minigrid-cu128.docker.tar.gz) |
| `objaverse_3dgen` | computer vision | `clip_score` | [objaverse_3dgen-cu118](./objaverse_3dgen-cu118.docker.tar.gz) | [objaverse_3dgen-cu128](./objaverse_3dgen-cu128.docker.tar.gz) |
| `ogbg_molpcba` | AI for science | `ap` | [ogbg_molpcba-cu118](./ogbg_molpcba-cu118.docker.tar.gz) | [ogbg_molpcba-cu128](./ogbg_molpcba-cu128.docker.tar.gz) |
| `qlib_stock` | AI for science | `ic_mean` | [qlib_stock-cu118](./qlib_stock-cu118.docker.tar.gz) | [qlib_stock-cu128](./qlib_stock-cu128.docker.tar.gz) |
| `qm9` | AI for science | `mae` | [qm9-cu118](./qm9-cu118.docker.tar.gz) | [qm9-cu128](./qm9-cu128.docker.tar.gz) |
| `screenspot_pro` | computer vision | `grounding_accuracy` | [screenspot_pro-cu118](./screenspot_pro-cu118.docker.tar.gz) | [screenspot_pro-cu128](./screenspot_pro-cu128.docker.tar.gz) |
| `split_cifar100` | computer vision | `accuracy` | [split_cifar100-cu118](./split_cifar100-cu118.docker.tar.gz) | [split_cifar100-cu128](./split_cifar100-cu128.docker.tar.gz) |
| `terra_incognita` | computer vision | `accuracy` | [terra_incognita-cu118](./terra_incognita-cu118.docker.tar.gz) | [terra_incognita-cu128](./terra_incognita-cu128.docker.tar.gz) |
| `vlabench_manipulation` | robotics | `success_rate` | [vlabench_manipulation-cu118](./vlabench_manipulation-cu118.docker.tar.gz) | [vlabench_manipulation-cu128](./vlabench_manipulation-cu128.docker.tar.gz) |
| `voicebank_demand` | audio/speech | `pesq` | [voicebank_demand-cu118](./voicebank_demand-cu118.docker.tar.gz) | [voicebank_demand-cu128](./voicebank_demand-cu128.docker.tar.gz) |
| `weatherbench_z500t850` | AI for science | `rmse` | [weatherbench_z500t850-cu118](./weatherbench_z500t850-cu118.docker.tar.gz) | [weatherbench_z500t850-cu128](./weatherbench_z500t850-cu128.docker.tar.gz) |
| `wilds_fmow` | AI for science | `worst_region_accuracy` | [wilds_fmow-cu118](./wilds_fmow-cu118.docker.tar.gz) | [wilds_fmow-cu128](./wilds_fmow-cu128.docker.tar.gz) |

## CUDA variants

- `*-cu118.docker.tar.gz` — built on `nvidia/cuda:11.8.0-runtime-ubuntu22.04`.
- `*-cu128.docker.tar.gz` — built on `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (e.g. RTX 5090).

Use `cu118` unless your GPU requires CUDA 12.x kernels. Both variants produce identical task behaviour and are interchangeable from the agent's point of view.

## License

The FARBench framework is released under the Apache-2.0 License. The bundled datasets and pre-cached model weights are redistributed from their original sources and retain the original licenses; see each task's README in the data repository.
