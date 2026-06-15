# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.dgxcloud import DGXCloudExecutor, DGXCloudState
from nemo_run.core.execution.launcher import SlurmTemplate

# Reuse the DGXCloud (Run:ai) torchx scheduler for our CLI subclass. The scheduler
# is selected by *exact* executor class (EXECUTOR_MAPPING[executor.__class__]), so a
# subclass must be registered explicitly or get_executor_str() raises KeyError.
from nemo_run.run.torchx_backend.schedulers.api import (
    EXECUTOR_MAPPING,
    REVERSE_EXECUTOR_MAPPING,
)


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME)
logger = logging.getLogger(__name__)

# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""

PERF_ENV_VARS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
    "TRANSFORMERS_OFFLINE": "1",  # Default for benchmark runs that mostly use NullTokenizer.
    "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
    "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
    "NVTE_NORM_FWD_USE_CUDNN": "1",
    "NVTE_NORM_BWD_USE_CUDNN": "1",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "HF_HUB_OFFLINE": "0",  # Keep Hub online by default; --offline flips this to 1.
}


@dataclass(kw_only=True)
class RunAICliExecutor(DGXCloudExecutor):
    """Run:ai executor that submits via the ``runai`` CLI instead of the REST API.

    Architecturally this mirrors the Slurm model (ambient user login + a submit
    binary) rather than the DGX Cloud REST model (machine-to-machine app_token):

      * ``DGXCloudExecutor`` authenticates with ``appId``/``appSecret`` (an admin-created
        Run:ai Application) and POSTs JSON to ``{base_url}/workloads/...``.
      * ``RunAICliExecutor`` reuses an interactive ``runai login`` (SSO) context — exactly
        like ``sbatch``/``srun`` rely on the user already being on a login node — and shells
        out to ``runai training pytorch submit``. **No Application / client credentials.**

    Everything else (PVC mounts, torchrun command materialization, code packaging onto the
    shared PVC-backed ``NEMORUN_HOME``) is inherited unchanged, so the same recipe code runs.

    The class is registered in NeMo-Run's ``EXECUTOR_MAPPING`` so ``run.run()`` routes it
    through the existing DGXCloud torchx scheduler, which only calls ``package()`` then
    ``launch(name, cmd)`` — both of which we satisfy here.
    """

    # Mayo Run:ai needs SR-IOV rails + a Multus network annotation for RoCE/GDR, which the
    # GCP-managed DGX Cloud REST path does not set. These are plumbed through as CLI flags.
    runai_extended_resources: list[str] = field(default_factory=list)  # e.g. ["nvidia.com/r0-p0=1", ...]
    runai_annotations: list[str] = field(default_factory=list)  # e.g. ["k8s.v1.cni.cncf.io/networks=..."]
    runai_rails_on_master: bool = True  # also emit --master-extended-resource for each rail
    runai_large_shm: bool = True
    runai_node_pools: Optional[str] = None
    runai_extra_submit_args: list[str] = field(default_factory=list)
    runai_print_only: bool = False  # build + print the command, do not submit

    # --- Auth: no token. Reuse the ambient `runai login` / kube context. ---
    def get_auth_token(self) -> Optional[str]:  # type: ignore[override]
        return "runai-cli"  # non-empty sentinel so any inherited guard passes

    def get_project_and_cluster_id(self, token: str):  # type: ignore[override]
        # The `runai` CLI resolves project/cluster from `-p <project>` + the logged-in
        # context, so we don't need to look up REST IDs.
        return (self.project_name or "default", "cli")

    def move_data(self, *args, **kwargs):  # type: ignore[override]
        # No REST data-mover: package() writes code directly into the PVC-backed
        # NEMORUN_HOME (launched_from_cluster semantics).
        return None

    def _runai_submit_argv(self, name: str) -> list[str]:
        workers = max(self.nodes - 1, 0)
        argv: list[str] = [
            "runai", "training", "pytorch", "submit", name,
            "-p", self.project_name,
            "-i", self.container_image,
            "-g", str(self.gpus_per_node),
            "--workers", str(workers),
        ]
        if self.runai_large_shm:
            argv.append("--large-shm")
        if self.runai_node_pools:
            argv += ["--node-pools", self.runai_node_pools]
        for pvc in self.pvcs:
            claim, path = pvc.get("claimName"), pvc.get("path")
            if claim and path:
                argv += ["--existing-pvc", f"claimname={claim},path={path}"]
        for res in self.runai_extended_resources:
            argv += ["--extended-resource", res]
            if self.runai_rails_on_master:
                argv += ["--master-extended-resource", res]
        for ann in self.runai_annotations:
            argv += ["--annotation", ann]
        for key, value in self.env_vars.items():
            if value is None:
                continue
            argv += ["-e", f"{key}={value}"]
        argv += self.runai_extra_submit_args
        argv += ["--command", "--", "/bin/bash", f"{self.pvc_job_dir}/launch_script.sh"]
        return argv

    def launch(self, name: str, cmd: list[str]) -> tuple[str, str]:  # type: ignore[override]
        name = name.replace("_", "-").replace(".", "-").lower()  # K8s name rules
        # In-pod bootstrap, identical to DGXCloudExecutor.launch (symlink /nemo_run, cd
        # into the staged code dir, tee per-rank logs into the PVC for fetch_logs()).
        launch_script = (
            f"\nln -s {self.pvc_job_dir}/ /nemo_run\n"
            f"cd /nemo_run/code\n"
            f"mkdir -p {self.pvc_job_dir}/logs\n"
            f'{" ".join(cmd)} 2>&1 | tee -a {self.pvc_job_dir}/log_$HOSTNAME.out '
            f"{self.pvc_job_dir}/log-allranks_0.out\n"
        )
        with open(os.path.join(self.job_dir, "launch_script.sh"), "w+") as f:
            f.write(launch_script)

        argv = self._runai_submit_argv(name)
        printable = " ".join(shlex.quote(a) for a in argv)
        logger.info("Run:ai CLI submit command:\n%s", printable)
        print(f"\n[RunAICliExecutor] {printable}\n")

        if self.runai_print_only:
            return name, "PrintOnly"

        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"`runai` submit failed (rc={result.returncode}).\n"
                f"CMD: {printable}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        logger.info("runai submit stdout:\n%s", result.stdout)
        return name, "Running"

    # --- Status / cancel via CLI (avoid REST). Logs reuse the parent's PVC tail. ---
    def status(self, job_id: str):  # type: ignore[override]
        return DGXCloudState("Running")

    def cancel(self, job_id: str):  # type: ignore[override]
        subprocess.run(
            ["runai", "training", "pytorch", "delete", job_id, "-p", self.project_name],
            capture_output=True,
            text=True,
        )


# Route RunAICliExecutor through the existing DGXCloud torchx scheduler (exact-class lookup).
EXECUTOR_MAPPING.setdefault(RunAICliExecutor, "dgx_cloud")
REVERSE_EXECUTOR_MAPPING.setdefault("runai_cli", RunAICliExecutor)

# Experiment.run() gates parallel/detach on *exact* class membership (uses ==, not isinstance),
# so a DGXCloudExecutor subclass is not recognized. Register RunAICliExecutor explicitly.
try:
    from nemo_run.run.experiment import Experiment as _NRExperiment

    if RunAICliExecutor not in _NRExperiment._PARALLEL_SUPPORTED_EXECUTORS:
        _NRExperiment._PARALLEL_SUPPORTED_EXECUTORS = (
            tuple(_NRExperiment._PARALLEL_SUPPORTED_EXECUTORS) + (RunAICliExecutor,)
        )
    if RunAICliExecutor not in _NRExperiment._DETACH_SUPPORTED_EXECUTORS:
        _NRExperiment._DETACH_SUPPORTED_EXECUTORS = (
            tuple(_NRExperiment._DETACH_SUPPORTED_EXECUTORS) + (RunAICliExecutor,)
        )
except Exception:  # pragma: no cover - best-effort registration
    pass


def slurm_executor(
    gpu: str,
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    offline: bool = False,
    nemo_home: str = DEFAULT_NEMO_HOME,
    wandb_key: str = None,
    network: str = None,
    custom_bash_cmds: List[List[str]] = None,
    additional_slurm_params: Dict[str, Any] = None,
    gres: Optional[str] = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments

    Args:
        additional_slurm_params: Dict[str, Any], optional
            Additional SLURM parameters to pass to sbatch. These will be converted to #SBATCH directives.
            Example: {"nodelist": "node001,node002", "constraint": "gpu"} will generate:
                #SBATCH --nodelist=node001,node002
                #SBATCH --constraint=gpu
    """
    custom_bash_cmds = [] if custom_bash_cmds is None else [" ".join(cmd) for cmd in custom_bash_cmds]
    mounts = []
    # Explicitly request GPU resources to ensure proper allocation
    # Without --gres=gpu:N, some clusters only allocate 1 GPU regardless of ntasks_per_node
    srun_args = custom_srun_args.copy() + [
        "--mpi=pmix",
        "--no-container-mount-home",
        "--container-writable",  # Required for benchmark compatibility on read-only-by-default container setups.
    ]

    if log_dir is not None:
        set_nemorun_home(log_dir)
    else:
        if os.environ.get("NEMORUN_HOME") is None:
            logger.warning(
                f"Logs will be written to {get_nemorun_home()}, which is probably not desired.  export NEMORUN_HOME in your shell environment or use the --log_dir argument"
            )

    if wandb_key is not None:
        PERF_ENV_VARS["WANDB_API_KEY"] = wandb_key

    if gpu.lower() == "gb200":
        PERF_ENV_VARS["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        PERF_ENV_VARS["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        PERF_ENV_VARS["NEMO_HOME"] = nemo_home
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        # Enable authenticated online access for tokenizer/config paths.
        PERF_ENV_VARS.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})
    if offline:
        # Disable HF Hub network calls. Requires a populated local HF cache.
        PERF_ENV_VARS["HF_HUB_OFFLINE"] = "1"

    PERF_ENV_VARS.update(custom_env_vars)
    mounts.extend(custom_mounts)

    # add --segment flag to sbatch if job uses GB200.
    segment = None
    if num_gpus_per_node == 4:
        if nodes <= 18:
            segment = nodes
        else:  # nodes > 18
            for segment_candidate in range(18, 0, -1):
                if nodes % segment_candidate == 0:
                    segment = segment_candidate
                    break

    numa_divisor = 2 if gpu.lower() in ["gb200", "gb300"] else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    custom_bash_cmds.append(numa_cmd)

    launcher = SlurmTemplate(
        template_inline=INLINE_TEMPLATE,
        template_vars={"pre_cmds": " ; ".join(custom_bash_cmds)},
    )

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        gres=gres,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=PERF_ENV_VARS,
        container_env=sorted(PERF_ENV_VARS.keys()),
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.Packager(),
        segment=segment,
        network=network,
        launcher=launcher,
        additional_parameters=additional_slurm_params,
    )

    return executor


def dgxc_executor(
    dgxc_base_url: str,
    dgxc_cluster: str,
    dgxc_kube_apiserver_url: str,
    dgxc_app_id: str,
    dgxc_app_secret: str,
    dgxc_project_name: str,
    dgxc_pvc_claim_name: str,
    nodes: int,
    num_gpus_per_node: int,
    wandb_key: str = None,
    hf_token: str = None,
    custom_env_vars: Dict[str, str] = None,
    dgxc_pvc_mount_path: str = "/nemo-workspace",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
):
    """
    DGXCloud cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """

    env_vars = {
        "TORCH_HOME": "/nemo-workspace/.cache",
        "FI_EFA_USE_HUGE_PAGE": "0",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "NCCL_TUNER_PLUGIN": "/opt/gcp-ofi-nccl/install/lib/libnccl-ofi-tuner.so",
        "WANDB_API_KEY": wandb_key,
        "HF_TOKEN": hf_token,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "False",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/nemo-workspace/pagaray/hf_cache",
    }
    if custom_env_vars:
        env_vars.update(custom_env_vars)
    executor = run.DGXCloudExecutor(
        base_url=dgxc_base_url,
        kube_apiserver_url=dgxc_kube_apiserver_url,
        app_id=dgxc_app_id,
        app_secret=dgxc_app_secret,
        project_name=dgxc_project_name,
        nodes=nodes,
        gpus_per_node=num_gpus_per_node,
        container_image=container_image,
        pvc_nemo_run_dir=get_nemorun_home(),
        launched_from_cluster=True,
        pvcs=[
            {
                "name": "workspace",
                "path": dgxc_pvc_mount_path,
                "existingPvc": True,
                "claimName": dgxc_pvc_claim_name,
            }
        ],
        custom_spec=(
            {
                "annotations": [{"name": "runai.dgxc.nvidia.com/gcp-nccl", "value": "none", "exclude": False}],
            }
            if dgxc_cluster == "dgxcloud-gcp" and nodes == 1
            else {}
        ),
        env_vars=env_vars,
        launcher="torchrun",
    )
    return executor


def runai_cli_executor(
    project_name: str,
    pvc_claim_name: str,
    nodes: int,
    num_gpus_per_node: int,
    container_image: str,
    pvc_mount_path: str = "/nemo-workspace",
    extended_resources: Optional[List[str]] = None,
    annotations: Optional[List[str]] = None,
    rails_on_master: bool = True,
    large_shm: bool = True,
    node_pools: Optional[str] = None,
    extra_submit_args: Optional[List[str]] = None,
    print_only: bool = False,
    wandb_key: str = None,
    hf_token: str = None,
    custom_env_vars: Dict[str, str] = None,
):
    """Run:ai executor that submits via the ``runai`` CLI (no Application credentials).

    This is the CLI analogue of :func:`dgxc_executor`. It builds the same Run:ai workload
    (PVC mounts, distributed PyTorch, torchrun) but submits with ``runai training pytorch
    submit`` instead of the REST API, so it only needs an interactive ``runai login``.

    The base env mirrors the authoritative post-PerfEnvPlugin set we validated on the Mayo
    B300 cluster, minus the GCP/AWS-isms baked into :func:`dgxc_executor`, and uses the new
    ``PYTORCH_ALLOC_CONF`` name plus ``NCCL_GRAPH_REGISTER=0`` (required with CUDA graphs +
    expandable_segments). The PerfEnvPlugin layers model-specific vars on top at setup.
    """
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NCCL_GRAPH_REGISTER": "0",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "TOKENIZERS_PARALLELISM": "False",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_API_KEY": wandb_key,
        "HF_TOKEN": hf_token,
    }

    # Slurm's srun exports the full submit-side environment into the container, so
    # launch.sh's `export HF_HOME=...` (the mounted HF cache) is honored automatically.
    # Run:ai only injects env vars we list explicitly, so forward HF_HOME/HF_HUB_CACHE
    # here; otherwise the container falls back to ~/.cache/huggingface and offline
    # tokenizer loads fail even though the cache PVC is mounted.
    _hf_home = os.environ.get("HF_HOME")
    if _hf_home:
        env_vars["HF_HOME"] = _hf_home
        env_vars["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")

    if custom_env_vars:
        env_vars.update(custom_env_vars)

    return RunAICliExecutor(
        # DGXCloudExecutor required fields; REST ones are unused by the CLI path.
        base_url="",
        app_id="",
        app_secret="",
        project_name=project_name,
        container_image=container_image,
        pvc_nemo_run_dir=get_nemorun_home(),
        launched_from_cluster=True,
        nodes=nodes,
        gpus_per_node=num_gpus_per_node,
        pvcs=[
            {
                "name": "workspace",
                "path": pvc_mount_path,
                "existingPvc": True,
                "claimName": pvc_claim_name,
            }
        ],
        env_vars=env_vars,
        launcher="torchrun",
        # Run:ai-CLI-specific extras
        runai_extended_resources=extended_resources or [],
        runai_annotations=annotations or [],
        runai_rails_on_master=rails_on_master,
        runai_large_shm=large_shm,
        runai_node_pools=node_pools,
        runai_extra_submit_args=extra_submit_args or [],
        runai_print_only=print_only,
    )


def get_executor(
    platform: str,
    *,
    gpu: str,
    num_gpus: int,
    gpus_per_node: int,
    log_dir: str,
    time_limit: str,
    container_image: str,
    custom_mounts: List[str],
    custom_env_vars: Dict[str, str],
    custom_srun_args: List[str],
    custom_bash_cmds: List[List[str]],
    hf_token: Optional[str],
    offline: bool,
    nemo_home: str,
    wandb_key: Optional[str],
    # Slurm-specific
    account: Optional[str] = None,
    partition: Optional[str] = None,
    additional_slurm_params: Optional[Dict[str, Any]] = None,
    gres: Optional[str] = None,
    # Run:ai / DGXCloud-specific
    dgxc_base_url: Optional[str] = None,
    dgxc_cluster: Optional[str] = None,
    dgxc_kube_apiserver_url: Optional[str] = None,
    dgxc_app_id: Optional[str] = None,
    dgxc_app_secret: Optional[str] = None,
    dgxc_project_name: Optional[str] = None,
    dgxc_pvc_claim_name: Optional[str] = None,
    dgxc_pvc_mount_path: str = "/nemo-workspace",
    # Run:ai CLI-specific (platform="runai"): no app credentials required
    runai_extended_resources: Optional[List[str]] = None,
    runai_annotations: Optional[List[str]] = None,
    runai_rails_on_master: bool = True,
    runai_large_shm: bool = True,
    runai_node_pools: Optional[str] = None,
    runai_extra_submit_args: Optional[List[str]] = None,
    runai_print_only: bool = False,
):
    """Factory that returns the NeMo-Run executor for the requested platform.

    This is the single dispatch point for scheduler backends so the rest of the
    benchmark code is platform-agnostic. Select with ``--platform`` (or the
    ``PLATFORM`` env var):

      - ``slurm`` -> :func:`slurm_executor`      (run.SlurmExecutor)
      - ``runai`` -> :func:`runai_cli_executor`  (RunAICliExecutor, ``runai`` CLI, SSO login, no app creds)
      - ``dgxc``  -> :func:`dgxc_executor`       (run.DGXCloudExecutor, Run:ai REST API, app_id/app_secret)
      - ``local`` -> run.LocalExecutor           (single node, torchrun)
    """
    platform = (platform or "slurm").lower()
    # ceil(num_gpus / gpus_per_node)
    nodes = -(num_gpus // -gpus_per_node)

    if platform == "slurm":
        for name, val in (("--account", account), ("--partition", partition)):
            if not val:
                raise ValueError(f"platform='slurm' requires {name}")
        return slurm_executor(
            gpu=gpu,
            account=account,
            partition=partition,
            log_dir=log_dir,
            nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            custom_mounts=custom_mounts,
            custom_env_vars=custom_env_vars,
            custom_srun_args=custom_srun_args,
            custom_bash_cmds=custom_bash_cmds,
            gres=gres,
            hf_token=hf_token,
            offline=offline,
            nemo_home=nemo_home,
            additional_slurm_params=additional_slurm_params,
            wandb_key=wandb_key,
        )

    if platform == "runai":
        # CLI path: needs only an interactive `runai login` + project/PVC. No app creds.
        required = {
            "--dgxc_project_name": dgxc_project_name,
            "--dgxc_pvc_claim_name": dgxc_pvc_claim_name,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"platform='runai' requires the following arguments: {', '.join(missing)}"
            )
        return runai_cli_executor(
            project_name=dgxc_project_name,
            pvc_claim_name=dgxc_pvc_claim_name,
            pvc_mount_path=dgxc_pvc_mount_path,
            nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            container_image=container_image,
            extended_resources=runai_extended_resources,
            annotations=runai_annotations,
            rails_on_master=runai_rails_on_master,
            large_shm=runai_large_shm,
            node_pools=runai_node_pools,
            extra_submit_args=runai_extra_submit_args,
            print_only=runai_print_only,
            custom_env_vars=custom_env_vars,
            wandb_key=wandb_key,
            hf_token=hf_token,
        )

    if platform == "dgxc":
        # REST path: requires a Run:ai Application (app_id/app_secret).
        # NOTE: dgxc_kube_apiserver_url is currently an unused field on
        # nemo_run.DGXCloudExecutor (submission goes through the REST base_url),
        # so it is intentionally not required here.
        required = {
            "--dgxc_base_url": dgxc_base_url,
            "--dgxc_app_id": dgxc_app_id,
            "--dgxc_app_secret": dgxc_app_secret,
            "--dgxc_project_name": dgxc_project_name,
            "--dgxc_pvc_claim_name": dgxc_pvc_claim_name,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"platform='dgxc' requires the following arguments: {', '.join(missing)}"
            )
        return dgxc_executor(
            dgxc_base_url=dgxc_base_url,
            # dgxc_cluster only toggles a GCP-specific annotation upstream; default to platform name.
            dgxc_cluster=dgxc_cluster or platform,
            dgxc_kube_apiserver_url=dgxc_kube_apiserver_url,
            dgxc_app_id=dgxc_app_id,
            dgxc_app_secret=dgxc_app_secret,
            dgxc_project_name=dgxc_project_name,
            dgxc_pvc_claim_name=dgxc_pvc_claim_name,
            dgxc_pvc_mount_path=dgxc_pvc_mount_path,
            custom_env_vars=custom_env_vars,
            nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            container_image=container_image,
            wandb_key=wandb_key,
            hf_token=hf_token,
        )

    if platform == "local":
        return run.LocalExecutor(launcher="torchrun", env_vars=custom_env_vars or {})

    raise ValueError(
        f"Unknown platform '{platform}'. Valid options: slurm, runai, dgxc, local."
    )
