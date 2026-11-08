# HairDeck on Modal

Runs HairDeck on a Modal GPU, so no local GPU is needed. Apache-2.0, like the rest of this repository.

## What it builds

- Image: Debian, Python 3.12, PyTorch 2.8.0 + torchvision 0.23.0 (CUDA 12.8 wheels), `requirements.txt`, `pyrender`, ffmpeg,
  the libraries headless Blender needs, and OSMesa for the previews (`HAIRDECK_GL=osmesa`).
- Volume `hairdeck-root` (mounted at `/vol/hairdeck`, which is `HAIRDECK_ROOT`): checkpoints, data, third-party weights,
  Blender 4.2.23 LTS with the VRM Add-on 4.10.2. `download_weights.py` skips what is there and checks sha256, so reruns do not download again.
- GPU: `L4` (24 GB) by default. Set `HAIRDECK_MODAL_GPU=A10G` to change it.

## Run

```bash
pip install modal && modal setup              # once
modal run modal/hairdeck_app.py::setup        # fills the volume (about 4 minutes, CPU only)
modal run modal/hairdeck_app.py::main --images my_images/ --out results/ --n 2
```

`main` uploads every `.png`/`.jpg` in `--images`, runs `scripts/image_to_hair.py` (`--n` variations, `--seed`, `--steps`, `--cfg`),
exports every variation as VRM with `hairdeck.vrm.export` (`--no-vrm` skips it, `--body BODY.json` picks another avatar body),
and unpacks the results under `results/out/<image name>/` with `modal_timings.json`.

## Gated weights

SAM 3 (`facebook/sam3`) and DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) are gated on Hugging Face. The token in the Modal
secret named by `HAIRDECK_HF_SECRET` (default `redsky-huggingface`, key `HF_TOKEN`) must belong to an account that accepted both licences.
`setup` first prints the HTTP status of both repositories for that token (200 = access, 403 = licence not accepted).
`setup --no-include-gated` does everything except those two downloads.

## Notes

- The code is mounted from this checkout at each run, so a change needs no image rebuild.
