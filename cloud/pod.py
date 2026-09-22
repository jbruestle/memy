"""RunPod pod management for memy. Needs RUNPOD_API_KEY in the environment.

  python cloud/pod.py gpus                       # GPU types + on-demand prices
  python cloud/pod.py create --gpu H100 --count 1 --volume <network-volume-id>
  python cloud/pod.py list
  python cloud/pod.py ssh <pod-id>               # print the ssh command
  python cloud/pod.py stop <pod-id>              # stops billing for GPU, keeps container disk
  python cloud/pod.py terminate <pod-id>         # deletes the pod (network volume survives)

The network volume (create it once in the RunPod console, in a data center
that lists H100/B200) mounts at /workspace and holds the venv, model,
datasets, llama.cpp and runs/, so a new pod only needs bootstrap.sh to
verify. Pod creation passes REPO_URL through to the pod environment.
"""

import argparse
import os
import sys

import runpod

DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def _key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("set RUNPOD_API_KEY")
    runpod.api_key = key


def cmd_gpus(args):
    _key()
    rows = []
    for g in runpod.get_gpus():
        info = runpod.get_gpu(g["id"])
        rows.append((g["id"], info.get("memoryInGb"), info.get("securePrice"), info.get("communityPrice")))
    rows.sort(key=lambda r: (r[2] or 1e9))
    print(f"{'gpu':40s} {'GB':>4s} {'secure $/h':>10s} {'community $/h':>13s}")
    for gid, mem, sp, cp in rows:
        if args.filter and args.filter.lower() not in gid.lower():
            continue
        print(f"{gid:40s} {mem or 0:4d} {sp or 0:10.2f} {cp or 0:13.2f}")


def _gpu_id(name):
    for g in runpod.get_gpus():
        if g["id"] == name:
            return g["id"]
    matches = [g["id"] for g in runpod.get_gpus() if name.lower() in g["id"].lower()]
    if len(matches) != 1:
        sys.exit(f"gpu '{name}' matches {matches}; be more specific")
    return matches[0]


def cmd_create(args):
    _key()
    gpu = _gpu_id(args.gpu)
    env = {"WORKSPACE": "/workspace"}
    if args.repo_url:
        env["REPO_URL"] = args.repo_url
    pod = runpod.create_pod(
        name=args.name, image_name=args.image, gpu_type_id=gpu, gpu_count=args.count,
        cloud_type=args.cloud, container_disk_in_gb=args.disk, volume_in_gb=0,
        network_volume_id=args.volume, volume_mount_path="/workspace",
        ports="22/tcp,8080/http", env=env, start_ssh=True, support_public_ip=True,
        min_vcpu_count=args.count * 8, min_memory_in_gb=args.count * 60,
        data_center_id=args.dc)
    print("created:", pod.get("id"), gpu, "x", args.count)
    print("next: python cloud/pod.py ssh", pod.get("id"))


def cmd_list(args):
    _key()
    for p in runpod.get_pods():
        m = p.get("machine") or {}
        rt = p.get("runtime") or {}
        print(f"{p['id']}  {p.get('name'):20s} {p.get('desiredStatus'):10s} "
              f"{p.get('gpuCount')}x {m.get('gpuDisplayName')}  uptime {rt.get('uptimeInSeconds', 0)}s  "
              f"${p.get('costPerHr', 0)}/h")


def _ssh(pod):
    rt = pod.get("runtime") or {}
    for port in rt.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return f"ssh root@{port['ip']} -p {port['publicPort']} -i ~/.ssh/id_ed25519"
    return None


def cmd_ssh(args):
    _key()
    pod = runpod.get_pod(args.pod_id)
    s = _ssh(pod)
    print(s or "no public ssh port yet (pod still starting?)")


def cmd_stop(args):
    _key()
    print(runpod.stop_pod(args.pod_id))


def cmd_terminate(args):
    _key()
    print(runpod.terminate_pod(args.pod_id) or "terminated")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpus"); g.add_argument("--filter", default=None); g.set_defaults(f=cmd_gpus)
    c = sub.add_parser("create")
    c.add_argument("--name", default="memy")
    c.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--volume", required=True, help="network volume id")
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--disk", type=int, default=50, help="container disk GB")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"])
    c.add_argument("--dc", default=None, help="data center id (must match the volume's)")
    c.add_argument("--repo-url", default=os.environ.get("REPO_URL"))
    c.set_defaults(f=cmd_create)
    sub.add_parser("list").set_defaults(f=cmd_list)
    s = sub.add_parser("ssh"); s.add_argument("pod_id"); s.set_defaults(f=cmd_ssh)
    s = sub.add_parser("stop"); s.add_argument("pod_id"); s.set_defaults(f=cmd_stop)
    s = sub.add_parser("terminate"); s.add_argument("pod_id"); s.set_defaults(f=cmd_terminate)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
