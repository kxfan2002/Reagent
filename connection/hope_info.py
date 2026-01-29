import os
import json

cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)
node_rank = cluster_spec["index"]
nnodes = len(cluster_spec[role])
nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
master = cluster_spec[role][0]
master_addr, master_ports = master.split(":")
master_ports = master_ports.split(",")

print('{} {}'.format(master_addr, node_rank))
print(master_ports[0])
print("--main_process_ip={} --main_process_port={}".format(master_addr, master_ports[0]))

