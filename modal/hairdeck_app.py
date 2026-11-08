"""HairDeck on Modal.

One Modal app that runs HairDeck (image -> stylized hair cards + spring-bone chains -> GLB / VRM) on a cloud GPU,
so no local GPU is needed. See modal/README.md.

    modal run modal/hairdeck_app.py::setup                      # once: fill the weights volume
    modal run modal/hairdeck_app.py::main --images DIR --out DIR --n 2

Environment (read on the machine that runs `modal`):
    HAIRDECK_MODAL_GPU     GPU type, default L4 (24 GB; A10G also fits)
    HAIRDECK_HF_SECRET     name of the Modal secret that holds HF_TOKEN, default redsky-huggingface. The account behind
                           the token must have accepted the licences of facebook/sam3 and
                           facebook/dinov3-vitl16-pretrain-lvd1689m.
"""
import io
import json
import os
import pathlib
import subprocess
import tarfile

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
GPU = os.environ.get("HAIRDECK_MODAL_GPU", "L4")
HF_SECRET = os.environ.get("HAIRDECK_HF_SECRET", "redsky-huggingface")
ROOT = "/vol/hairdeck"  # HAIRDECK_ROOT: checkpoints, data, third-party weights, Blender (the volume)
CODE = "/opt/hairdeck"  # the repository

APT = [
    "ffmpeg", "git", "xz-utils", "ca-certificates", "curl",
    # Blender 4.2 headless
    "libx11-6", "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libxkbcommon0", "libsm6", "libice6",
    "libgl1", "libegl1", "libxext6", "libgomp1",
    # pyrender previews without a display
    "libosmesa6", "libglu1-mesa",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*APT)
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128",
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_file(str(REPO / "requirements.txt"), "/tmp/requirements.txt", copy=True)
    .run_commands("pip install -r /tmp/requirements.txt", "pip install --no-deps pyrender==0.1.45")
    .env({
        "HAIRDECK_ROOT": ROOT,
        "HAIRDECK_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "HF_HOME": f"{ROOT}/hf_cache",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        str(REPO), CODE,
        ignore=["**/.git/**", "**/__pycache__/**", "results/**", "modal/inputs/**", "modal/out/**"],
    )
)

volume = modal.Volume.from_name("hairdeck-root", create_if_missing=True)
app = modal.App("hairdeck", image=image)
secrets = [modal.Secret.from_name(HF_SECRET)]


def _sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=CODE, **kw)


@app.function(volumes={ROOT: volume}, secrets=secrets, timeout=3600, cpu=4)
def download(only: list[str]):
    """Fill the volume with `scripts/setup/download_weights.py --only ...` (skips what is present, checks sha256)."""
    _sh(["python", "scripts/setup/download_weights.py", "--only", *only])
    volume.commit()


@app.function(volumes={ROOT: volume}, secrets=secrets, timeout=600)
def gated_access() -> dict:
    """Does the HF token reach the two gated repositories? Prints HTTP statuses only, never the token."""
    import urllib.error
    import urllib.request

    tok = os.environ.get("HF_TOKEN", "")
    out = {}
    for name, url in {
        "facebook/sam3": "https://huggingface.co/facebook/sam3/resolve/main/config.json",
        "facebook/dinov3-vitl16-pretrain-lvd1689m":
            "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/config.json",
    }.items():
        req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out[name] = r.status
        except urllib.error.HTTPError as e:
            out[name] = e.code
    print(out)
    return out


@app.function(gpu=GPU, volumes={ROOT: volume}, secrets=secrets, timeout=3600, cpu=4, memory=24576)
def generate(images: dict[str, bytes], n: int = 2, seed: int = 0, vrm: bool = True, body: str | None = None,
             steps: int = 50, cfg: float = 2.0) -> bytes:
    """images: {file name: bytes}. Runs image_to_hair.py, then the VRM export of every variation, and returns a tar of
    the results (OUT/<image name>/...). `body` is a body JSON in the workspace (default data/avatar/body.json)."""
    import time

    volume.reload()
    work = pathlib.Path("/tmp/job")
    (work / "in").mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        (work / "in" / name).write_bytes(data)
    out = work / "out"
    t0 = time.time()
    _sh(["python", "scripts/image_to_hair.py", *map(str, sorted((work / "in").iterdir())), "--out", str(out),
         "--n", str(n), "--seed", str(seed), "--steps", str(steps), "--cfg", str(cfg)])
    t_gen = time.time() - t0
    vrm_log = {}
    if vrm:
        t1 = time.time()
        for d in sorted(p for p in out.iterdir() if p.is_dir()):
            for s in range(seed, seed + n):
                if not (d / f"cards_seed{s}.json").exists():
                    continue
                cmd = ["python", "-m", "hairdeck.vrm.export", str(d), str(s), "vrm", "--no-render"]
                if body:
                    cmd += ["--body", body]
                try:
                    _sh(cmd)
                    vrm_log[f"{d.name}/{s}"] = "ok"
                except subprocess.CalledProcessError as e:
                    vrm_log[f"{d.name}/{s}"] = f"failed: {e}"
        t_vrm = time.time() - t1
    else:
        t_vrm = 0.0
    (out / "modal_timings.json").write_text(json.dumps(dict(gpu=GPU, generate_s=t_gen, vrm_s=t_vrm, vrm=vrm_log), indent=1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname="out")
    return buf.getvalue()


@app.function(volumes={ROOT: volume}, timeout=3600, cpu=4, memory=16384)
def export_on_body(results_tar: bytes, names: list[str], vrm: bytes, hair_glb: bytes, key: str, h: int = 100,
                   seeds: list[int] | None = None) -> bytes:
    """Export saved results (the tar `generate` returned) as VRM on another VRoid body, with no new diffusion run.
    vrm: the body's VRM 0.x; hair_glb: the body's own hair as a GLB in the glTF axes (Y up, +Z front), in metres;
    key: the body's name (also the file tag). Builds the body JSON (`hairdeck.vrm.body build`), then exports the
    variations of `names` (folders under out/). Returns a tar of out/<name>/<key>_seed<s>.vrm and the body JSON."""
    volume.reload()
    work = pathlib.Path("/tmp/body")
    (work / "hair").mkdir(parents=True, exist_ok=True)
    (work / f"{key}.vrm").write_bytes(vrm)
    (work / "hair" / "hair.glb").write_bytes(hair_glb)
    with tarfile.open(fileobj=io.BytesIO(results_tar), mode="r:gz") as tar:
        tar.extractall(work, filter="data")
    body_json = work / f"{key}.json"
    _sh(["python", "-m", "hairdeck.vrm.body", "build", str(work / f"{key}.vrm"), str(work / "hair"), key, str(h), key,
         str(body_json)])
    res = {}
    for name in names:
        d = work / "out" / name
        for sd in sorted(d.glob("cards_seed*.json")):
            s = int(sd.stem.removeprefix("cards_seed"))
            if seeds is not None and s not in seeds:
                continue
            try:
                _sh(["python", "-m", "hairdeck.vrm.export", str(d), str(s), "vrm", "--no-render", "--body", str(body_json)])
                res[f"{name}/{s}"] = "ok"
            except subprocess.CalledProcessError as e:
                res[f"{name}/{s}"] = f"failed: {e}"
    (work / "export_log.json").write_text(json.dumps(res, indent=1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(body_json, arcname=f"{key}.json")
        tar.add(work / "export_log.json", arcname="export_log.json")
        for name in names:
            for f in (work / "out" / name).glob(f"{key}_seed*.vrm"):
                tar.add(f, arcname=f"{name}/{f.name}")
            for f in (work / "out" / name).glob("vrm_seed*/*_export.json"):
                tar.add(f, arcname=f"{name}/{f.parent.name}_{f.name}")
    return buf.getvalue()


@app.function(volumes={ROOT: volume}, timeout=1800, cpu=4)
def run_shell(cmd: str) -> str:
    """Debug helper: run a shell command in the image with the volume mounted."""
    volume.reload()
    r = subprocess.run(cmd, shell=True, cwd=CODE, capture_output=True, text=True)
    return r.stdout + r.stderr


@app.local_entrypoint()
def setup(include_gated: bool = True):
    """Download the weights into the volume. The gated items need an HF account that accepted both licences."""
    print(gated_access.remote())
    download.remote(["hairdeck", "birefnet", "sam3_src", "dinov3_src", "blender"])
    if include_gated:
        download.remote(["sam3", "dinov3"])


@app.local_entrypoint()
def main(images: str, out: str = "modal/out", n: int = 2, seed: int = 0, vrm: bool = True, body: str = "",
         steps: int = 50, cfg: float = 2.0):
    """images: a folder of images. out: where the results unpack."""
    files = {p.name: p.read_bytes() for p in sorted(pathlib.Path(images).iterdir())
             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    print(f"{len(files)} images, n={n}, gpu={GPU}")
    blob = generate.remote(files, n=n, seed=seed, vrm=vrm, body=body or None, steps=steps, cfg=cfg)
    dest = pathlib.Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print("results in", dest / "out")
