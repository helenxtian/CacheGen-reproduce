"""Builds cachegen_fig9_table1.ipynb, the Trovi notebook.

Usage (repository root): python scripts/reproduce/build_notebook.py cachegen_fig9_table1.ipynb
Edit this file, not the notebook, and re-run it.
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Reproducing CacheGen (SIGCOMM '24): Figure 9 and Table 1

This notebook reproduces two results from *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving* (Liu et al., SIGCOMM 2024) on Chameleon Cloud:

- **Figure 9**: KV cache size vs. quality for CacheGen and the uniform-quantization baseline.
- **Table 1**: Mistral-7B on LongChat, comparing 8-bit quantization (paper: 622 MB, accuracy 1.00) with CacheGen (paper: 176 MB, 0.98).

It also runs our reimplementation of the token-wise **delta encoding** from §5.2 of the paper ([src/delta_codec.py](src/delta_codec.py)). The authors' released codec does not include it.

**What the notebook does:**
1. Reserves one bare-metal GPU node and a floating IP.
2. Launches `CC-Ubuntu24.04-CUDA` on the node.
3. Installs the authors' conda environment and builds the CUDA arithmetic coder.
4. Finds a disk with enough space for the KV caches, mounting the node's spare disk if it needs to.
5. Runs a 3-document smoke test, then the full experiment, in the background on the node.
6. Downloads the figure, the table, and the raw result JSONs into this Jupyter workspace.
7. Checks the results against the paper's claims and prints PASS or FAIL for each one.

**Before you start:**
- Open this notebook from Chameleon's JupyterHub (for example via Trovi). The `chi` library and SSH keys are configured automatically there.
- Your project needs an allocation with GPU node hours.
- **Time:**
  - setup: about 30 minutes;
  - smoke test: 30 to 60 minutes;
  - full 50-document run: several hours.

  The run time scales with `END`, the number of documents. If you're short on time, set `END = 20`; the checks in step 8 still apply.

**Hardware:** any NVIDIA GPU with at least 16 GB works. The paper used A40/A100 GPUs; we tested this notebook on:
- a Tesla V100-SXM2-32GB (CHI@UC, `gpu_v100`), which produced the committed results;
- a Tesla P100-SXM2-16GB (CHI@TACC, `gpu_p100_nvlink`).
- an NVIDIA A100 on the GigaIO node `gigaio05` (CHI@TACC), the default in step 1. If it's taken, set `NODE_NAME = None` and pick a `NODE_TYPE` that's free.

The GPU affects speed, not KV cache sizes. Quality can differ by a borderline answer or two between GPUs.

**Expected result** (Mistral-7B, 50 LongChat documents):

| Technique | Paper (MB / accuracy) | Our run (MB / accuracy) |
|---|---|---|
| 8-bit quantization | 622 / 1.00 | 587 / 1.00 |
| CacheGen (level 2) | 176 / 0.98 | 173 / 1.00 |

That is a 3.4x reduction; the paper reports 3.5x. Our sizes are a few percent smaller because these 50 documents average 8,935 tokens. All of our results and their analysis are in `reproduce-results/README.md`.
""")

md(r"""
## 0. Our results (no GPU needed)

This cell shows the results committed in `reproduce-results/`, so you can see what the run produces before reserving anything. It reads only files in this folder. The run in steps 1-8 regenerates these numbers on your own node.
""")

code(r'''
import os
from IPython.display import Image, Markdown, display

if os.path.isdir("reproduce-results"):
    display(Markdown("**Table 1, our run (Mistral-7B, 50 LongChat documents, 1x V100):**"))
    display(Markdown(open("reproduce-results/table1_longchat.md").read()))
    display(Markdown("**Figure 9, LongChat (broken x-axis):**"))
    display(Image("reproduce-results/figures/fig9_break.png"))
    display(Markdown("**Figure 9, all three datasets, including our extension levels (section 9):**"))
    display(Image("reproduce-results/figures/fig9_all_datasets.png"))
    print("Details and analysis: reproduce-results/README.md")
else:
    print("reproduce-results/ is not next to this notebook; open the notebook from the repository folder.")
''')

md(r"""
## 1. Configuration

Edit these values, then run every cell from top to bottom. Every step is safe to re-run: an existing lease, server, environment or KV cache is reused rather than created again.

> **Pick a node that really has a GPU.**
> - Some GigaIO nodes at CHI@TACC have no GPU, or their hardware inventory is out of date.
> - Use https://discover.chameleoncloud.org and the site's lease calendar to find a GPU node that's free for the next day.
> - Check the GPU check in step 4 before running anything long.
>
> **Key pairs are per site:** your key pair must exist at the site you pick.
""")

code(r'''
# --- Chameleon resources ---
SITE        = "CHI@TACC"
NODE_TYPE   = "gpu_a100"        # used only if NODE_NAME is None. Also tested: "gpu_v100" (CHI@UC) and
                                # "gpu_p100_nvlink" (CHI@TACC); any >= 16 GB NVIDIA GPU should work
NODE_NAME   = "gigaio05"        # a GigaIO node with one A100; not every GigaIO node has a GPU. None = any NODE_TYPE node
LEASE_NAME  = "cachegen-fig9"   # an existing ACTIVE lease with this name is reused; give yours a unique name
                                # if you share a project, so you don't pick up someone else's lease
LEASE_HOURS = 24
SERVER_NAME = "cachegen-fig9"   # likewise reused if it exists; make it unique in a shared project
IMAGE       = "CC-Ubuntu24.04-CUDA"   # needs the NVIDIA driver; do not use the vGPU image on bare metal

# --- Code ---
REPO_URL    = "https://github.com/helenxtian/CacheGen-reproduce.git"
REPO_BRANCH = "main"
REPO_DIR    = "~/CacheGen-reproduce"

# --- Experiment ---
DATASET   = "longchat"    # Table 1 uses LongChat
START     = 0
END       = 50            # documents in the full run (our committed results use 50); 20 is about 40% of the time
SMOKE_END = 3             # documents in the smoke test

# Passed as environment variables to scripts/reproduce/fig9_table1.sh
RUN_ENV = {
    "LONGCHAT_JUDGE":  "string",   # "string" = substring match, no API key; "gpt" needs OPENAI_API_KEY
    "CACHEGEN_LEVELS": "1 2 3",    # CacheGen encoding levels (points on the Fig. 9 curve)
    "QUANT_BITS":      "3 4 8",    # uniform-quantization baselines
    "DELTA_ANCHORS":   "level",    # our delta codec; "level int8" runs both anchor modes, "" skips it
    "KEEP_ENCODED":    "0",        # delete each codec run's encoded files after scoring (saves ~50 GB)
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",   # less fragmentation; needed on 16 GB GPUs
}
SAVE_DIR = "auto"     # where raw KV caches go (~1.2 GB per document). "auto" = step 4b picks a big disk
XBREAK = (310, 520)   # Figure 9 x-axis break in MB: detailed panel up to 310, then from 520 (8-bit, fp16)
SMOKE_ENV = dict(RUN_ENV, CACHEGEN_LEVELS="2", QUANT_BITS="8", RESULTS_DIR="smoke_results")
''')

md(r"""
## 2. Select project and site

Run the next cell, pick your project in the dropdown, **then** run the cell after it.
""")

code(r'''
import datetime, io, json, os, shlex, tarfile, time
from IPython.display import Image, Markdown, clear_output, display
from chi import context, lease, server

context.version = "1.0"
context.choose_project()
''')

code(r'''
context.use_site(SITE)
print("Using site", SITE)
''')

md(r"""
## 3. Reserve a GPU node and launch the instance

The lease asks for **one physical host and one floating IP**. The cell refuses to continue if the lease has no `physical:host` reservation. A lease without one is the most common cause of *"No valid host was found"* when launching.
""")

code(r'''
l = lease.Lease(LEASE_NAME, duration=datetime.timedelta(hours=LEASE_HOURS))
if NODE_NAME:
    l.add_node_reservation(amount=1, node_name=NODE_NAME)
else:
    l.add_node_reservation(amount=1, node_type=NODE_TYPE)
l.add_fip_reservation(amount=1)
try:
    l.submit(wait_for_active=True, idempotent=True, show="text")   # reuses a non-terminated lease with this name
except Exception as e:
    raise RuntimeError(f"Could not create lease {LEASE_NAME}: {e}\nUsually no {NODE_NAME or NODE_TYPE} node is free "
                       f"at {SITE} right now. Check the host calendar on the site's Leases page in the dashboard, "
                       "or pick another NODE_TYPE/SITE in step 1.") from e
l.refresh()

if l.status != "ACTIVE":
    raise RuntimeError(f"Lease {LEASE_NAME} is {l.status}. If it is ERROR, the node is probably taken: "
                       "delete the lease, change NODE_TYPE/NODE_NAME, and re-run.")
if not l.node_reservations:
    raise RuntimeError(f"Lease {LEASE_NAME} has no physical:host reservation (only floating IPs). "
                       "Delete it in the GUI and re-run this cell.")
reservation_id = l.node_reservations[0]["id"]
print("Lease:", l.name, l.id)
print("Host reservation:", reservation_id)
print("Ends:", l.end_date, "UTC")
''')

code(r'''
existing = [s_ for s_ in server.list_servers() if s_.name == SERVER_NAME]
if existing:
    print(f"Reusing server {SERVER_NAME} (status {existing[0].status})")
else:
    server.Server(SERVER_NAME, reservation_id=reservation_id, image_name=IMAGE).submit(
        wait_for_active=True, show="text")    # bare metal takes about 10-20 minutes

s = server.get_server(SERVER_NAME)
if s.status == "ERROR":
    try:     # Nova's reason for the failure, e.g. "No valid host was found"
        from chi.clients import nova
        fault = (getattr(nova().servers.get(s.id), "fault", None) or {}).get("message", "unknown")
    except Exception as e:
        fault = f"could not read it ({e})"
    raise RuntimeError(f"The server is in ERROR state. Nova's reason: {fault}\n"
                       "If it is 'No valid host was found', another instance may already be using the reserved "
                       "node (for example one launched from the dashboard on this lease): delete it. Then run "
                       "server.delete_server(s.id), wait a minute, and re-run this cell.")

fip = s.get_floating_ip()
if not fip:
    reserved = l.get_reserved_floating_ips()   # empty if the lease has no floating-IP reservation
    server.associate_floating_ip(s.id, reserved[0] if reserved else None)
    # The server's address list can take a minute to show the new IP; SSH needs it there
    for _ in range(20):
        s = server.get_server(SERVER_NAME)
        fip = s.get_floating_ip()
        if fip:
            break
        time.sleep(15)
    if not fip:
        raise RuntimeError("The floating IP was attached but never appeared on the server. Re-run this cell; "
                           "if it happens again, check Network > Floating IPs in the dashboard.")
s.check_connectivity(show="text")

# Port 22 can open a little before the node accepts our key, so retry the login a few times
for attempt in range(6):
    try:
        with s.ssh_connection() as c:
            c.run("true", hide=True)
        break
    except Exception as e:
        if attempt == 5:
            raise RuntimeError(
                f"Port 22 on {fip} is open, but SSH login failed ({type(e).__name__}: {e}). If {SERVER_NAME} was "
                "launched from the dashboard with your own key pair, this notebook can't log in to it: delete "
                f"that server (server.delete_server(s.id)), wait a minute, and re-run this cell so the notebook "
                "launches one with your JupyterHub key.") from e
        time.sleep(30)
print(f"Server is up at {fip}. From a Jupyter terminal: ssh cc@{fip}")
''')

md(r"""
Two small helpers to run commands on the node and copy files back. Output streams into the cell.
""")

code(r'''
def run(cmd, hide=False, warn=False):
    """Run a shell command on the node (as user cc)."""
    with s.ssh_connection() as c:
        return c.run(cmd, hide=hide, warn=warn, pty=False, in_stream=False)


def put_text(text, remote_path):
    with s.ssh_connection() as c:
        c.put(io.BytesIO(text.encode()), remote_path)


def get(remote_path, local_path):
    with s.ssh_connection() as c:
        c.get(remote_path, local_path)
''')

md(r"""
## 4. Check the GPU

Stop here if this fails. A bare-metal node without a visible NVIDIA GPU can't run the experiment, and no driver install will change that.
""")

code(r'''
r = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", warn=True, hide=True)
if not r.ok or not r.stdout.strip():
    run("lspci | grep -iE 'nvidia|vga|3d' || true")
    raise RuntimeError("No NVIDIA GPU/driver on this node. If lspci shows no NVIDIA device, the node has no GPU: "
                       "delete the server and lease and reserve a different node.")
print(r.stdout)
run("df -h ~ | tail -1")
''')

md(r"""
### 4b. Find disk space for the KV caches

The full run stores about 1.2 GB of raw KV cache per document, plus one codec run's encoded files at a time. The model (about 15 GB) and the conda environment also go on the root disk.

If the root disk has too little free space, this cell uses the node's largest unused disk instead:
- If the disk is already formatted, it mounts the disk.
- If the disk has no filesystem at all, it formats it, then mounts it.

It never touches a disk that is mounted or that has another kind of filesystem.
""")

code(r'''
need_gb = int(1.5 * END) + 30
PREP_DISK = f"""
set -e
NEED={need_gb}
avail() {{ df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }}
if [ "$(avail $HOME)" -ge $NEED ]; then echo "SAVE_DIR=$HOME"; exit 0; fi
if mountpoint -q /mnt/data && [ "$(avail /mnt/data)" -ge $NEED ]; then echo "SAVE_DIR=/mnt/data"; exit 0; fi
dev=$(python3 - <<'PY'
import json, subprocess
tree = json.loads(subprocess.check_output(["lsblk", "-J", "-b", "-o", "NAME,PATH,SIZE,TYPE,MOUNTPOINT,FSTYPE"]))
def used(d):
    return bool(d.get("mountpoint")) or any(used(c) for c in d.get("children") or [])
cands = []
for d in tree["blockdevices"]:
    if d["type"] != "disk" or used(d):
        continue
    kids = d.get("children") or []
    cands += [d] if not kids else [c for c in kids if c["type"] == "part"]
cands = [c for c in cands if c.get("fstype") in (None, "ext4", "xfs") and int(c["size"]) > {need_gb} * 1e9]
print(max(cands, key=lambda c: int(c["size"]))["path"] if cands else "")
PY
)
if [ -z "$dev" ]; then
    echo "No unused disk with {need_gb} GB found; free space on the root disk or lower END." >&2
    echo "SAVE_DIR=$HOME"; exit 0
fi
if [ -z "$(lsblk -no FSTYPE $dev)" ]; then echo "Formatting empty disk $dev" >&2; sudo mkfs.ext4 -q -F $dev; fi
sudo mkdir -p /mnt/data && sudo mount $dev /mnt/data && sudo chown cc:cc /mnt/data
echo "Mounted $dev at /mnt/data" >&2
echo "SAVE_DIR=/mnt/data"
"""
if SAVE_DIR == "auto":
    put_text(PREP_DISK, "prep_disk.sh")
    out = run("bash prep_disk.sh", hide="stdout").stdout
    SAVE_DIR = [l_ for l_ in out.splitlines() if l_.startswith("SAVE_DIR=")][-1].split("=", 1)[1]
run(f"df -h {SAVE_DIR} | tail -1")
print(f"KV caches go to SAVE_DIR={SAVE_DIR} (need about {need_gb} GB for END={END})")
''')

md(r"""
## 5. Install the software (about 20-30 minutes, the first time only)

This runs the README's install steps:
1. Install Miniconda.
2. Clone the repository.
3. Create the `cachegen` conda environment from `env.yaml`. It pins torch 2.3.1, transformers 4.42.3 and fschat 0.2.36, and includes the CUDA 12 compiler.
4. Install LMCache.
5. Build the `torchac_cuda` arithmetic-coding kernel.
""")

code(r'''
SETUP = f"""
set -e
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
if [ ! -x ~/miniconda3/bin/conda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
fi
source ~/miniconda3/etc/profile.d/conda.sh
for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
    conda tos accept --override-channels --channel $ch >/dev/null 2>&1 || true
done

if [ ! -d {REPO_DIR} ]; then
    git clone -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}
else
    git -C {REPO_DIR} pull --ff-only
fi
cd {REPO_DIR}
git log -1 --format='code version: %h %s'

if ! conda env list | grep -q '^cachegen '; then
    conda env create -f env.yaml
fi
conda activate cachegen
pip install -q -e LMCache
if ! python -c 'import torchac_cuda' 2>/dev/null; then
    (cd LMCache/third_party/torchac_cuda && python setup.py install)
fi
python -c 'import torch, torchac_cuda; print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))'
"""
put_text(SETUP, "setup_cachegen.sh")
run("bash setup_cachegen.sh")
''')

md(r"""
## 6. Run the experiment in the background

`scripts/reproduce/fig9_table1.sh` does five steps:
1. Prefill each document and save its raw fp16 KV cache.
2. CacheGen at encoding levels 1-3.
3. Our delta-encoding codec.
4. Uniform quantization at 3, 4 and 8 bits.
5. The uncompressed baseline.

Each step writes a JSON file to the results folder.

Jobs run detached on the node, so closing this notebook or losing the connection doesn't stop them. `wait_job` only watches the log. If you interrupt it, the job keeps going; re-run the cell to resume watching.
""")

code(r'''
def launch(tag, start, end, env):
    if run("pgrep -f '[f]ig9_table1.sh'", warn=True, hide=True).ok:
        print("A fig9_table1.sh job is already running; not starting another. Use wait_job() to watch it.")
        return
    exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in dict(env, SAVE_DIR=SAVE_DIR).items())
    # All-or-nothing chain (a failed conda activate must not run the script in the wrong env);
    # its output, including setup errors, goes to the log, and its exit code to <tag>.done
    inner = (f"( source ~/miniconda3/etc/profile.d/conda.sh && conda activate cachegen && cd {REPO_DIR} && "
             f"export CUDA_VISIBLE_DEVICES=0 {exports} && "
             f"bash scripts/reproduce/fig9_table1.sh {DATASET} {start} {end} ) > ~/logs/{tag}.log 2>&1; "
             f"echo $? > ~/logs/{tag}.done")
    run(f"mkdir -p ~/logs && rm -f ~/logs/{tag}.done && "
        f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &")
    print(f"Started '{tag}' (docs {start}-{end}). Log on the node: ~/logs/{tag}.log")


def wait_job(tag, poll=60, tail=20):
    t0 = time.time()
    while True:
        try:
            done = run(f"cat ~/logs/{tag}.done 2>/dev/null", warn=True, hide=True)
            log = run(f"tail -n {tail} ~/logs/{tag}.log 2>/dev/null", warn=True, hide=True).stdout
        except Exception as e:      # transient SSH hiccup: retry
            print("ssh error, retrying:", e); time.sleep(poll); continue
        clear_output(wait=True)
        print(f"[{tag}] watching for {(time.time() - t0) / 60:.0f} min\n" + log)
        if done.ok and done.stdout.strip():
            code_ = int(done.stdout.strip())
            print(f"[{tag}] finished with exit code {code_}")
            if code_ != 0:
                raise RuntimeError(f"Job {tag} failed; see the full log with run('less ~/logs/{tag}.log') "
                                   f"or run('tail -n 200 ~/logs/{tag}.log')")
            return
        time.sleep(poll)
''')

md(r"""
### 6a. Smoke test (3 documents, level 2 / 8-bit only)

This checks the whole pipeline before you spend hours on the full run: model download, prefill, the CUDA coder, the delta codec, and evaluation. Results go to `smoke_results/` so they don't mix with the real ones.
""")

code(r'''
launch("smoke", START, SMOKE_END, SMOKE_ENV)
wait_job("smoke")
run(f"cd {REPO_DIR} && ~/miniconda3/envs/cachegen/bin/python scripts/reproduce/plot_fig9.py --results_dir smoke_results --out smoke_fig9.png")
''')

md(r"""
**Automatic smoke check.** Before you spend hours on the full run, the next cell checks two things:
- every method produced a result;
- CacheGen level 2 is several times smaller than 8-bit quantization. The paper reports about 3.5x.

With only 3 documents, accuracies are noisy, so the cell doesn't check them.
""")

code(r'''
def node_results(results_dir):
    """All result JSONs of DATASET in results_dir on the node, keyed by file name without .json."""
    py = ("import json,glob,os;print(json.dumps({os.path.basename(f)[:-5]: json.load(open(f)) "
          f"for f in glob.glob('{results_dir}/{DATASET}/*.json')}}))")
    return json.loads(run(f"cd {REPO_DIR} && ~/miniconda3/envs/cachegen/bin/python -c {shlex.quote(py)}",
                          hide=True).stdout)

smoke = node_results("smoke_results")
expected = ["cachegen_level2", "quant_8bit", "vanilla_fp16"] + (["cachegen_delta_level2"] if SMOKE_ENV.get("DELTA_ANCHORS") else [])
missing = [k for k in expected if k not in smoke]
if missing:
    raise RuntimeError(f"Smoke test is missing {missing}; look at run('tail -n 100 ~/logs/smoke.log')")
for k in expected:
    print(f"{k:<24} {smoke[k]['avg_size_mb']:8.1f} MB   quality {smoke[k]['avg_metric']:.2f}   ({smoke[k]['num_docs']} docs)")
ratio = smoke["quant_8bit"]["avg_size_mb"] / smoke["cachegen_level2"]["avg_size_mb"]
print(f"\n8-bit / CacheGen level 2 = {ratio:.2f}x (paper: 3.5x)")
if not 2.5 <= ratio <= 4.5:
    raise RuntimeError("The compression ratio is far from the paper's; check the smoke log before the full run.")
print("Smoke test OK: continue with the full run.")
''')

md(r"""
### 6b. Full run

Raw KV caches from the smoke test are reused, so those 3 documents aren't prefilled again.
""")

code(r'''
launch("full", START, END, RUN_ENV)
wait_job("full", poll=300)
''')

md(r"""
## 7. Collect results

This step uses the same layout as the repository's `reproduce-results/` folder.
1. On the node, it assembles `reproduce-results-new/`:
   - figures: broken-axis, log-scale and linear Figure 9, plus the smoke-test figure
   - Table 1 and a text table of every point
   - the per-method JSONs
   - logs, and a hardware/version record
2. It downloads that folder into this Jupyter workspace. Files under `~/work` persist after the lease ends.
3. It compares your numbers with the committed run.

The output goes to `reproduce-results-new/`, so the committed `reproduce-results/` stays untouched for comparison.
""")

code(r'''
COLLECT = f"""
set -e
cd {REPO_DIR}
PY=~/miniconda3/envs/cachegen/bin/python
OUT=~/reproduce-results-new
rm -rf $OUT && mkdir -p $OUT/figures $OUT/data $OUT/logs
cp -r mistral7b_results/{DATASET} $OUT/data/{DATASET}
if [ -d smoke_results/{DATASET} ]; then cp -r smoke_results/{DATASET} $OUT/data/smoke_{DATASET}; fi
$PY scripts/reproduce/plot_fig9.py --results_dir $OUT/data --out $OUT/figures/fig9_break.png --xbreak {XBREAK[0]} {XBREAK[1]} > /dev/null
$PY scripts/reproduce/plot_fig9.py --results_dir $OUT/data --out $OUT/figures/fig9_log.png --logx > /dev/null
$PY scripts/reproduce/plot_fig9.py --results_dir $OUT/data --out $OUT/figures/fig9.png | sed '1,2d' > $OUT/fig9_table.txt
if [ -f smoke_fig9.png ]; then cp smoke_fig9.png $OUT/figures/; fi
$PY scripts/reproduce/make_table1.py --results_dir $OUT/data --dataset {DATASET} > /dev/null
mv $OUT/data/table1_{DATASET}.md $OUT/
cp ~/logs/*.log ~/setup_cachegen.sh $OUT/logs/ 2>/dev/null || true
{{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
   git log -1 --format='code: %H %s'
   echo "site: {SITE}, lease: {LEASE_NAME}, collected: $(date -u +%FT%TZ)"; }} > $OUT/hardware.txt
tar czf ~/reproduce-results-new.tgz -C ~ reproduce-results-new
"""
put_text(COLLECT, "collect_results.sh")
run("bash collect_results.sh")

get("reproduce-results-new.tgz", "reproduce-results-new.tgz")
with tarfile.open("reproduce-results-new.tgz") as t:
    t.extractall(".")
NEW = "reproduce-results-new"
print("Saved to", os.path.abspath(NEW))
print(open(f"{NEW}/hardware.txt").read())

display(Image(f"{NEW}/figures/fig9_break.png"))
display(Markdown(open(f"{NEW}/table1_{DATASET}.md").read()))
''')

code(r'''
# Compare with the committed run in reproduce-results/ (if this notebook sits in the repository)
OLD = f"reproduce-results/data/{DATASET}"
if os.path.isdir(OLD):
    print(f"{'result':<26} {'committed MB':>12} {'new MB':>8} {'committed q':>12} {'new q':>6}")
    for f in sorted(os.listdir(f"{NEW}/data/{DATASET}")):
        new = json.load(open(f"{NEW}/data/{DATASET}/{f}"))
        old = json.load(open(f"{OLD}/{f}")) if os.path.exists(f"{OLD}/{f}") else None
        print(f"{f[:-5]:<26} {old['avg_size_mb'] if old else float('nan'):>12.1f} {new['avg_size_mb']:>8.1f} "
              f"{old['avg_metric'] if old else float('nan'):>12.2f} {new['avg_metric']:>6.2f}")
else:
    print("No committed reproduce-results/ found next to this notebook; skipping the comparison.")
''')

md(r"""
## 8. Did it reproduce?

The next cell checks the paper's claims for Table 1 and Figure 9 against the results you just downloaded, and prints PASS or FAIL for each:

| Claim | Check |
|---|---|
| 1 | CacheGen (level 2) is about 3.5x smaller than 8-bit quantization (Table 1) |
| 2 | Its size, and 8-bit's, are within 15% of the paper's |
| 3 | CacheGen keeps the quality of 8-bit quantization (Table 1: 0.98 vs. 1.00) |
| 4 | Figure 9: at every level, CacheGen is smaller than 4-bit quantization and at least as accurate |
| 5 | Figure 9: uniform quantization loses quality first: 3-bit is larger than CacheGen level 1, yet no more accurate |

Quality comparisons allow 0.05 of slack, for borderline answers.
""")

code(r'''
R = {f[:-5]: json.load(open(f"{NEW}/data/{DATASET}/{f}")) for f in os.listdir(f"{NEW}/data/{DATASET}")}
size = lambda k: R[k]["avg_size_mb"]
qual = lambda k: R[k]["avg_metric"]
TOL = 0.05
PAPER = {"quant_8bit": 622, "cachegen_level2": 176}
levels = [k for k in ("cachegen_level1", "cachegen_level2", "cachegen_level3") if k in R]
need = ["cachegen_level1", "cachegen_level2", "quant_3bit", "quant_4bit", "quant_8bit"]
if any(k not in R for k in need):
    raise RuntimeError(f"Missing results {[k for k in need if k not in R]}; did the full run finish?")

ratio = size("quant_8bit") / size("cachegen_level2")
checks = [
    ("Table 1: CacheGen L2 is ~3.5x smaller than 8-bit (3.0-4.0x)", 3.0 <= ratio <= 4.0,
     f"{ratio:.2f}x ({size('cachegen_level2'):.0f} vs. {size('quant_8bit'):.0f} MB; paper 176 vs. 622)"),
    ("Table 1: sizes within 15% of the paper", all(abs(size(k) / v - 1) <= 0.15 for k, v in PAPER.items()),
     ", ".join(f"{k} {size(k) / v - 1:+.0%}" for k, v in PAPER.items())),
    ("Table 1: CacheGen L2 keeps 8-bit's quality", qual("cachegen_level2") >= qual("quant_8bit") - TOL,
     f"{qual('cachegen_level2'):.2f} vs. {qual('quant_8bit'):.2f} (paper 0.98 vs. 1.00)"),
    ("Fig. 9: every CacheGen level beats 4-bit (smaller, as accurate)",
     all(size(k) < size("quant_4bit") and qual(k) >= qual("quant_4bit") - TOL for k in levels),
     "; ".join(f"{k[-6:]} {size(k):.0f} MB/{qual(k):.2f}" for k in levels) + f"; 4-bit {size('quant_4bit'):.0f} MB/{qual('quant_4bit'):.2f}"),
    ("Fig. 9: 3-bit is larger than CacheGen L1 yet no more accurate",
     size("quant_3bit") > size("cachegen_level1") and qual("quant_3bit") <= qual("cachegen_level1"),
     f"3-bit {size('quant_3bit'):.0f} MB/{qual('quant_3bit'):.2f} vs. L1 {size('cachegen_level1'):.0f} MB/{qual('cachegen_level1'):.2f}"),
]
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
n_ok = sum(ok for _, ok, _ in checks)
print(f"\n{n_ok}/{len(checks)} checks passed on {R['cachegen_level2']['num_docs']} documents.")
''')

md(r"""
### 8b. Optional: re-score with the paper's GPT judge

Steps 6-8 score LongChat with a substring match: is the right topic named in the answer? The authors' artifact instead asked a GPT model whether each answer is correct. This cell re-scores your saved answers that way, with the artifact's exact prompt. It doesn't re-run the model.

**Requirements and cost:**
- An OpenAI API key, in `OPENAI_API_KEY`. Without one, the cell only counts the answers.
- About 100 distinct answers to judge, so a few cents.

The labels and the model's answers are sent to OpenAI. The artifact used `gpt-3.5-turbo`; the default here is `gpt-4o-mini`.
""")

code(r'''
import subprocess, sys
# os.environ["OPENAI_API_KEY"] = "sk-..."   # or set it in your Jupyter environment
GPT_MODEL = "gpt-4o-mini"

log = f"{NEW}/logs/full.log"
cmd = [sys.executable, "scripts/reproduce/rejudge_longchat.py", "--log", log,
       "--out_dir", f"{NEW}/gpt_judge", "--model", GPT_MODEL]
if not os.environ.get("OPENAI_API_KEY"):
    cmd.append("--dry_run")
    print("No OPENAI_API_KEY set: counting answers only (no API calls).\n")
print(subprocess.run(cmd, capture_output=True, text=True).stdout)
''')

md(r"""
### Notes for the report

- **Judge:** LongChat accuracy uses a substring match (`LONGCHAT_JUDGE=string`) by default. The paper's artifact used a GPT judge, which you can enable with `LONGCHAT_JUDGE=gpt` and an `OPENAI_API_KEY`.
- **Sizes:**
  - CacheGen sizes are the length of the encoded bitstream.
  - Quantization sizes are computed from the bit width.
  - The fp16 baseline is the size of the raw KV tensor.
- **Not reproduced:** the H2O and LLMLingua rows of Table 1 aren't part of the authors' artifact, so they aren't run here.
- **Delta encoding** (`cachegen_delta`) is our reimplementation of §5.2, not the authors' code. See the docstring in `src/delta_codec.py` for the design choices.
  - In our runs it was 9-21% *larger* than the released codec on all three datasets.
  - `scripts/reproduce/analyze_delta.py` shows why: neighbouring tokens in Mistral-7B's KV cache are nearly independent channel by channel. Undoing RoPE doesn't help.
  - See `reproduce-results/README.md`.
- **Record your hardware:** the GPU model printed in step 4 and the code version printed in step 5.
""")

md(r"""
## 9. Optional: our extensions (NarrativeQA, TriviaQA, lower levels, context length)

You don't need this section to reproduce Table 1 or the LongChat panel of Figure 9. It runs `scripts/reproduce/extensions.sh` and adds the following:

| Job | What it runs |
|---|---|
| `lowbit` | LongChat at CacheGen levels 0/-1/-2 and 2-bit quantization, plus the delta-encoding (RoPE) analysis. It reuses the full run's KV caches, so it is quick. |
| `nqa`, `tqa` | Figure 9's other two datasets, 50 documents each: every CacheGen level, the delta codec, 2/3/4/8-bit quantization and fp16 |
| `context` | The same 30 NarrativeQA questions at 10K, 20K and 30K tokens. Needs a GPU with at least 40 GB, e.g. an A100: 20K tokens did not fit on our 16 GB P100. |

With `EXT_JOBS = "auto"`, the cell picks the jobs from the node's GPUs:
- `context` is added only if every GPU has at least 40 GB.
- With one GPU (e.g. a GigaIO A100 node), all jobs run one after the other on it.
- With more GPUs, they run in parallel, one lane per GPU.

**Time:** each job takes a few hours (`lowbit` less), so a 1-GPU chain of all four can take most of a day. Check that your lease lasts long enough, or list fewer jobs, e.g. `EXT_JOBS = "lowbit+context"`.

**Disk:** up to about 140 GB at a time, in the same `SAVE_DIR` as the full run.

Our results are in part 2 of `reproduce-results/README.md`.
""")

code(r'''
import re
RUN_EXTENSIONS = False   # set to True to run this section
EXT_JOBS = "auto"        # or e.g. "lowbit+context" (one GPU, in order) or "nqa tqa" (two GPUs in parallel)
EXT_RESTART = False      # True = start the jobs again even if they already ran (e.g. after a failure)

def pick_ext_jobs():
    mem = [int(x) for x in run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
                               hide=True).stdout.split()]
    jobs = ["lowbit", "nqa", "tqa"] + (["context"] if min(mem) >= 40000 else [])
    n = len(mem)
    if n == 1:
        return "+".join(jobs)                       # one lane: everything in order on the only GPU
    lanes = [[] for _ in range(min(n, len(jobs)))]  # spread jobs over the GPUs, one lane each
    for i, j in enumerate(jobs):
        lanes[i % len(lanes)].append(j)
    return " ".join("+".join(l_) for l_ in lanes)

def ext_exit_codes(members):
    """{job: exit code} for the jobs that have finished (their ~/logs/ext_<job>.done exists)."""
    out = run("for j in " + " ".join(members) + "; do f=~/logs/ext_$j.done; "
              "[ -s $f ] && echo \"EXITCODE $j $(head -1 $f)\"; done; true", hide=True, warn=True).stdout
    return {j: int(c) for j, c in re.findall(r"^EXITCODE (\w+) (\d+)\s*$", out, re.M)}

def wait_ext(jobs, poll=600):
    members = jobs.replace("+", " ").split()
    t0 = time.time()
    while True:
        try:
            st = run(f"cd {REPO_DIR} && JOBS={shlex.quote(jobs)} bash scripts/reproduce/extensions.sh status",
                     hide=True, warn=True).stdout
            codes = ext_exit_codes(members)
        except Exception as e:
            print("ssh error, retrying:", e); time.sleep(poll); continue
        clear_output(wait=True)
        print(f"[{jobs}] watching for {(time.time() - t0) / 60:.0f} min\n{st}")
        if all(j in codes for j in members):
            failed = [j for j in members if codes[j] != 0]
            if failed:
                raise RuntimeError(f"{failed} failed; see run('tail -n 100 ~/logs/ext_{failed[0]}.log')")
            print("All extension jobs finished.")
            return
        time.sleep(poll)

if RUN_EXTENSIONS:
    jobs = pick_ext_jobs() if EXT_JOBS == "auto" else EXT_JOBS
    members = jobs.replace("+", " ").split()
    print("Extension jobs:", jobs)
    # Re-running this cell only resumes watching: it starts nothing while jobs run or after they finished
    running = run("pgrep -f '[e]xtensions.sh (lowbit|nqa|tqa|context)$'", warn=True, hide=True).ok
    finished = all(j in ext_exit_codes(members) for j in members)
    if running:
        print("Extension jobs are already running; watching them.")
    elif finished and not EXT_RESTART:
        print("These jobs already finished; run the collect cell below (or set EXT_RESTART = True).")
    else:
        run(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate cachegen && cd {REPO_DIR} && "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAVE_DIR={SAVE_DIR} JOBS={shlex.quote(jobs)} "
            f"bash scripts/reproduce/extensions.sh start")
    wait_ext(jobs)       # safe to interrupt: the jobs keep running on the node; re-run this cell to watch again
else:
    print("Skipped (RUN_EXTENSIONS = False).")
''')

code(r'''
# Collect the extension results (also works part-way through: it packs whatever has finished)
if RUN_EXTENSIONS:
    run(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate cachegen && cd {REPO_DIR} && "
        f"bash scripts/reproduce/extensions.sh collect > /dev/null")
    get("extensions_results.tgz", "extensions_results.tgz")
    with tarfile.open("extensions_results.tgz") as t:
        t.extractall(".")
    print("Saved to", os.path.abspath("extensions_results"))
    for fig in ("fig9_all_datasets.png", "context_length.png"):
        path = f"extensions_results/figures/{fig}"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            display(Image(path))
    for tab in ("fig9_all_datasets_table.txt", "context_length_table.txt"):
        path = f"extensions_results/{tab}"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(open(path).read())
''')

md(r"""
## 10. Clean up

Delete the server and the lease when you're done so they stop using your allocation. **Download the results first (step 7, and step 9 if you ran it).** Everything on the node is lost when it's deleted.
""")

code(r'''
CONFIRM_DELETE = False   # set to True and re-run to delete the server and the lease

if CONFIRM_DELETE:
    server.delete_server(s.id)
    l.delete()
    print("Deleted server and lease.")
else:
    print("Nothing deleted. Set CONFIRM_DELETE = True to release the node.")
''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote", sys.argv[1], len(cells), "cells")
