# === Cost-Aware Parallel SFC Placement using Multi-Stage Graphs ===

import networkx as nx
import random
import pandas as pd
from itertools import product, permutations

NUM_NODES = 15
NUM_SFC_REQUESTS = 10
PARALLEL_UNITS_RANGE = (2, 4)
MAX_VNFS_PER_UNIT = 3
MAX_CPU_PER_NODE = 10
NODE_ACTIVATION_COST = 2
PE_DEPLOY_UNIT_COST = 1

class PhysicalNode:
    def __init__(self, node_id, cpu_capacity):
        self.id = node_id
        self.cpu = cpu_capacity
        self.used_cpu = 0
        self.activated = False

    def has_resources(self, required):
        return (self.cpu - self.used_cpu) >= required

    def reserve(self, required):
        if not self.activated:
            self.activated = True
        self.used_cpu += required

    def reset(self):
        self.used_cpu = 0
        self.activated = False

class SFCRequest:
    def __init__(self, request_id, src, dest, stages, bw, max_delay):
        self.id = request_id
        self.src = src
        self.dest = dest
        self.stages = stages
        self.bw = bw
        self.max_delay = max_delay

def generate_physical_graph():
    G = nx.erdos_renyi_graph(NUM_NODES, 0.25, seed=42)
    for u, v in G.edges():
        G[u][v]['bandwidth'] = 100
        G[u][v]['delay'] = random.randint(1, 10)
        G[u][v]['cost'] = random.randint(1, 5)
    return G

def generate_sfc_requests(G, num_requests):
    requests = []
    for i in range(num_requests):
        src, dst = random.sample(list(G.nodes), 2)
        num_units = random.randint(*PARALLEL_UNITS_RANGE)
        stages = [random.randint(1, 3) for _ in range(num_units)]
        delay_budget = random.randint(20, 40)
        bw_demand = random.randint(10, 30)
        requests.append(SFCRequest(i, src, dst, stages, bw_demand, delay_budget))
    return requests

def construct_msg(G, sfc, physical_nodes, used_nodes=set()):
    MSG = nx.DiGraph()
    stage_nodes = []
    for stage_idx, cpu_req in enumerate(sfc.stages):
        current_stage = []
        for node in physical_nodes:
            if node.has_resources(cpu_req) and node.id not in used_nodes:
                vname = f"{node.id}_s{stage_idx}"
                MSG.add_node(vname, physical=node.id, stage=stage_idx, cpu=cpu_req)
                current_stage.append(vname)
        stage_nodes.append(current_stage)
    for i in range(len(stage_nodes) - 1):
        for u in stage_nodes[i]:
            for v in stage_nodes[i + 1]:
                pu, pv = MSG.nodes[u]['physical'], MSG.nodes[v]['physical']
                if pu == pv:
                    continue
                if nx.has_path(G, pu, pv):
                    path = nx.shortest_path(G, pu, pv, weight='delay')
                    delay = sum(G[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
                    cost = sum(G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
                    MSG.add_edge(u, v, delay=delay, cost=cost)
    return MSG, stage_nodes

def find_feasible_path(MSG, stage_nodes, sfc, G):
    paths = []
    for path in product(*stage_nodes):
        delay = 0
        cost = 0
        valid = True
        physical_seen = set()
        for node in path:
            pnode = MSG.nodes[node]['physical']
            if pnode in physical_seen:
                valid = False
                break
            physical_seen.add(pnode)
        if not valid:
            continue
        for i in range(len(path)-1):
            if MSG.has_edge(path[i], path[i+1]):
                delay += MSG[path[i]][path[i+1]]['delay']
                cost += MSG[path[i]][path[i+1]]['cost']
            else:
                valid = False
                break
        if not valid:
            continue
        total_delay = delay
        if total_delay <= sfc.max_delay:
            paths.append((path, total_delay, cost))
    return min(paths, key=lambda x: x[2]) if paths else None

def run_brute_force(G, sfc):
    all_nodes = list(G.nodes)
    placements = []
    for perm in permutations(all_nodes, len(sfc.stages)):
        valid = True
        total_delay = 0
        total_cost = 0
        for i in range(len(perm)-1):
            if not nx.has_path(G, perm[i], perm[i+1]):
                valid = False
                break
            path = nx.shortest_path(G, perm[i], perm[i+1], weight='delay')
            total_delay += sum(G[path[j]][path[j+1]]['delay'] for j in range(len(path)-1))
            total_cost += sum(G[path[j]][path[j+1]]['cost'] for j in range(len(path)-1))
        if not valid or total_delay > sfc.max_delay:
            continue
        placements.append((perm, total_delay, total_cost))
    return min(placements, key=lambda x: x[2]) if placements else None

def simulate_with_msg_and_brute():
    G = generate_physical_graph()
    sfc_requests = generate_sfc_requests(G, NUM_SFC_REQUESTS)
    physical_nodes = [PhysicalNode(i, MAX_CPU_PER_NODE) for i in range(NUM_NODES)]
    placements = []

    for sfc in sfc_requests:
        print(f"\nProcessing SFC-{sfc.id}: src={sfc.src}, dest={sfc.dest}, stages={sfc.stages}, max_delay={sfc.max_delay}")

        MSG_active, stage_nodes_active = construct_msg(G, sfc, physical_nodes)
        result_active = find_feasible_path(MSG_active, stage_nodes_active, sfc, G)
        if result_active is None:
            print(f"❌ No valid active path found for SFC-{sfc.id} in MSG")
            continue
        path_active, delay_active, cost_active = result_active
        used_nodes = set(MSG_active.nodes[p]['physical'] for p in path_active)
        MSG_backup, stage_nodes_backup = construct_msg(G, sfc, physical_nodes, used_nodes)
        result_backup = find_feasible_path(MSG_backup, stage_nodes_backup, sfc, G)

        if result_backup:
            path_backup, delay_backup, cost_backup = result_backup
            backup_delay_cost = 0
            for a_node, b_node in zip(path_active, path_backup):
                a_physical = MSG_active.nodes[a_node]['physical']
                b_physical = MSG_backup.nodes[b_node]['physical']
                if nx.has_path(G, a_physical, b_physical):
                    path = nx.shortest_path(G, a_physical, b_physical, weight='delay')
                    backup_delay_cost += sum(G[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
        else:
            path_backup, delay_backup, cost_backup, backup_delay_cost = [], 0, 0, 0

        total_cost_msg = cost_active + cost_backup + backup_delay_cost

        result_brute = run_brute_force(G, sfc)
        if result_brute:
            path_brute, delay_brute, cost_brute = result_brute
        else:
            path_brute, delay_brute, cost_brute = [], 0, 0

        placements.append({
            "SFC_ID": sfc.id,
            "Source": sfc.src,
            "Destination": sfc.dest,
            "MSG_Active_Path": path_active,
            "MSG_Backup_Path": path_backup,
            "MSG_Total_Cost": total_cost_msg,
            "MSG_Active_Delay": delay_active,
            "MSG_Backup_Delay": delay_backup,
            "BruteForce_Cost": cost_brute,
            "BruteForce_Delay": delay_brute
        })

    df = pd.DataFrame(placements)
    print("\nComparison Results (MSG vs Brute Force):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    simulate_with_msg_and_brute()
