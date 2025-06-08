# === Cost-Aware Parallel SFC Placement using Multi-Stage Graphs ===
import networkx as nx
import random
import pandas as pd
from itertools import product

# === Configuration ===
NUM_PHYSICAL_NODES = 6  # Adjust as needed for your topology
NUM_SFC_REQUESTS = 3
MAX_PE_STAGES = 4
MAX_CPU_PER_NODE = 7
NODE_ACTIVATION_COST = 2  # cost for activating a physical node
PE_DEPLOY_UNIT_COST = 1   # cost per unit CPU used for deployment

# === Physical Node Class ===
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

# === SFC Request Class ===
class SFCRequest:
    def __init__(self, request_id, src, dest, stages, bw, max_delay):
        self.id = request_id
        self.src = src
        self.dest = dest
        self.stages = stages
        self.bw = bw
        self.max_delay = max_delay

# === Check for matplotlib availability for visualization ===
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

def generate_physical_graph(num_nodes):
    """
    Generate a physical network graph with nodes and edges.
    Args:
        num_nodes: Number of nodes to generate (note: edges may reference more nodes).
    Returns:
        G: NetworkX graph with nodes and edges.
    """
    G = nx.Graph()
    # Edges may reference nodes beyond num_nodes; adjust if needed.
    edges = [
        (0, 1, {'bandwidth': 100, 'delay': 2, 'cost': 3}),
        (1, 2, {'bandwidth': 100, 'delay': 3, 'cost': 2}),
        (2, 3, {'bandwidth': 100, 'delay': 1, 'cost': 4}),
        (3, 4, {'bandwidth': 100, 'delay': 5, 'cost': 6}),
        (1, 5, {'bandwidth': 100, 'delay': 2, 'cost': 3}),
        (1, 4, {'bandwidth': 100, 'delay': 4, 'cost': 5}),
        (0, 5, {'bandwidth': 100, 'delay': 6, 'cost': 6}),
        (2, 6, {'bandwidth': 100, 'delay': 3, 'cost': 3}),
        (5, 7, {'bandwidth': 100, 'delay': 4, 'cost': 4}),
        (6, 7, {'bandwidth': 100, 'delay': 2, 'cost': 2}),
    ]
    G.add_edges_from(edges)

    for u, v in G.edges():
        print(f"Edge ({u}, {v}) → cost: {G[u][v]['cost']}, delay: {G[u][v]['delay']}")

    if HAS_PLT:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        edge_labels = { (u, v): f"c:{G[u][v]['cost']}, d:{G[u][v]['delay']}" for u, v in G.edges() }
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Physical Network Graph (cost c, delay d)")
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not found. Skipping graph visualization.")

    return G

def generate_sfc_requests(num_requests, physical_nodes):
    """
    Generate SFC requests for simulation.
    Args:
        num_requests: Number of SFC requests to generate.
        physical_nodes: List of PhysicalNode objects.
    Returns:
        List of SFCRequest objects.
    """
    # Example requests (customize as needed)
    sfc_requests = [
        SFCRequest(0, physical_nodes[0], physical_nodes[3], [2, 2, 1], 20, 20),
        SFCRequest(1, physical_nodes[1], physical_nodes[3], [2, 2, 1], 30, 23)
    ]
    # Ensure at least 'num_requests' are generated
    while len(sfc_requests) < num_requests:
        src, dst = random.sample(physical_nodes, 2)
        stages = [random.randint(1, 3) for _ in range(random.randint(2, MAX_PE_STAGES))]
        bw = random.randint(10, 30)
        max_delay = random.randint(20, 40)
        sfc_requests.append(SFCRequest(len(sfc_requests), src, dst, stages, bw, max_delay))
    return sfc_requests

def construct_msg(Gp, sfc, physical_nodes, used_nodes=set()):
    """
    Construct a Multi-Stage Graph (MSG) for SFC placement.
    Args:
        Gp: Physical network graph.
        sfc: SFC request.
        physical_nodes: List of PhysicalNode objects.
        used_nodes: Set of node IDs to exclude (for backup).
    Returns:
        MSG: Multi-stage graph.
        stage_nodes: List of node labels per stage.
    """
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
                if nx.has_path(Gp, pu, pv):
                    path = nx.shortest_path(Gp, pu, pv, weight='delay')
                    delay = sum(Gp[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
                    cost = sum(Gp[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
                    # Check bandwidth constraint (optional, adjust as needed)
                    # Here, we assume all link bandwidths >= sfc.bw (as per your setup)
                    MSG.add_edge(u, v, delay=delay, cost=cost)
    return MSG, stage_nodes

def find_feasible_path(MSG, stage_nodes, sfc, Gp):
    """
    Find the minimum-cost feasible path through the MSG.
    Args:
        MSG: Multi-stage graph.
        stage_nodes: List of node labels per stage.
        sfc: SFC request.
        Gp: Physical network graph.
    Returns:
        Tuple (path, total_delay, total_cost) if feasible, else None.
    """
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
        # Add source-to-first and last-to-destination path if needed
        first_physical = MSG.nodes[path[0]]['physical']
        src_delay = src_cost = 0
        if first_physical != sfc.src.id and nx.has_path(Gp, sfc.src.id, first_physical):
            src_path = nx.shortest_path(Gp, sfc.src.id, first_physical, weight='delay')
            src_delay = sum(Gp[src_path[i]][src_path[i+1]]['delay'] for i in range(len(src_path)-1))
            src_cost = sum(Gp[src_path[i]][src_path[i+1]]['cost'] for i in range(len(src_path)-1))
        last_physical = MSG.nodes[path[-1]]['physical']
        dest_delay = dest_cost = 0
        if last_physical != sfc.dest.id and nx.has_path(Gp, last_physical, sfc.dest.id):
            dest_path = nx.shortest_path(Gp, last_physical, sfc.dest.id, weight='delay')
            dest_delay = sum(Gp[dest_path[i]][dest_path[i+1]]['delay'] for i in range(len(dest_path)-1))
            dest_cost = sum(Gp[dest_path[i]][dest_path[i+1]]['cost'] for i in range(len(dest_path)-1))
        total_delay = delay + src_delay + dest_delay
        total_cost = cost + src_cost + dest_cost
        if total_delay <= sfc.max_delay:
            paths.append((path, total_delay, total_cost))
    return min(paths, key=lambda x: x[2]) if paths else None

def reserve_resources(path, MSG, physical_nodes):
    """
    Reserve resources for a path and calculate extra cost.
    Args:
        path: List of node labels in the path.
        MSG: Multi-stage graph.
        physical_nodes: List of PhysicalNode objects.
    Returns:
        Extra cost (node activation + CPU usage).
    """
    extra_cost = 0
    for node_label in path:
        node_id = MSG.nodes[node_label]['physical']
        cpu_req = MSG.nodes[node_label]['cpu']
        node = physical_nodes[node_id]
        if not node.activated:
            extra_cost += NODE_ACTIVATION_COST
        extra_cost += cpu_req * PE_DEPLOY_UNIT_COST
        node.reserve(cpu_req)
    return extra_cost

def run_monitoring_placement(placements, Gp, physical_nodes):
    """
    Place Virtual Monitoring Functions (VMFs) on all nodes used in active/backup paths.
    Args:
        placements: List of placement dictionaries.
        Gp: Physical network graph.
        physical_nodes: List of PhysicalNode objects.
    Returns:
        DataFrame of VMF placements.
    """
    print("\n--- VMF Placement Phase ---")
    vmf_placements = []
    vmf_id_counter = 0
    vmf_nodes = set()

    for placement in placements:
        all_paths = placement['Active_Path'] + placement['Backup_Path']
        physical_path = [int(p.split('_')[0]) for p in all_paths]
        for node in physical_path:
            if node not in vmf_nodes:
                vmf_nodes.add(node)
                vmf_placements.append({
                    'VMF_ID': vmf_id_counter,
                    'Host_Node': node,
                    'Monitors_SFC': placement['SFC_ID']
                })
                vmf_id_counter += 1

    print("VMF Placement Summary:")
    for v in vmf_placements:
        print(f"VMF-{v['VMF_ID']} placed at Node {v['Host_Node']} for SFC-{v['Monitors_SFC']}")

    return pd.DataFrame(vmf_placements)

def run_simulation():
    """
    Main simulation loop for SFC placement and monitoring.
    Returns:
        DataFrame of placement results.
    """
    print("\nGenerating Physical Network...")
    Gp = generate_physical_graph(NUM_PHYSICAL_NODES)
    physical_nodes = [PhysicalNode(i, MAX_CPU_PER_NODE) for i in range(NUM_PHYSICAL_NODES)]
    print("\nGenerating SFC Requests...")
    sfc_requests = generate_sfc_requests(NUM_SFC_REQUESTS, physical_nodes)
    for sfc in sfc_requests:
        print(f"SFC-{sfc.id}: src={sfc.src.id}, dest={sfc.dest.id}, stages={sfc.stages}, max_delay={sfc.max_delay}")
    placements = []
    for sfc in sfc_requests:
        print(f"\nPlacing SFC-{sfc.id}: src={sfc.src.id}, dest={sfc.dest.id}, stages={sfc.stages}, max_delay={sfc.max_delay}")
        MSG_active, stage_nodes_active = construct_msg(Gp, sfc, physical_nodes)
        if not all(stage_nodes_active):
            print("  ❌ No feasible placement for active path.")
            continue
        result_active = find_feasible_path(MSG_active, stage_nodes_active, sfc, Gp)
        if result_active is None:
            print("  ❌ No valid active path within delay constraint.")
            continue
        path_active, delay_active, cost_active = result_active
        print(f"  ✅ Active Path: [SRC {sfc.src.id}] → {path_active} → [DEST {sfc.dest.id}], Delay: {delay_active}, Cost: {cost_active} (includes entry/exit paths)")
        extra_cost_a = reserve_resources(path_active, MSG_active, physical_nodes)

        used_nodes = set(int(n.split('_')[0]) for n in path_active)
        MSG_backup, stage_nodes_backup = construct_msg(Gp, sfc, physical_nodes, used_nodes)
        result_backup = find_feasible_path(MSG_backup, stage_nodes_backup, sfc, Gp)
        if result_backup:
            path_backup, delay_backup, cost_backup = result_backup
            print(f"  ✅ Backup Path: [SRC {sfc.src.id}] → {path_backup} → [DEST {sfc.dest.id}], Delay: {delay_backup}, Cost: {cost_backup} (includes entry/exit paths)")
            extra_cost_b = reserve_resources(path_backup, MSG_backup, physical_nodes)
            # Calculate backup delay cost (C4)
            backup_delay_cost = 0
            for a_node, b_node in zip(path_active, path_backup):
                a_physical = MSG_active.nodes[a_node]['physical']
                b_physical = MSG_backup.nodes[b_node]['physical']
                if nx.has_path(Gp, a_physical, b_physical):
                    path = nx.shortest_path(Gp, a_physical, b_physical, weight='delay')
                    backup_delay_cost += sum(Gp[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
            total_cost = cost_active + cost_backup + extra_cost_a + extra_cost_b + backup_delay_cost
        else:
            print("  ❌ No valid backup path found.")
            path_backup, delay_backup, cost_backup = [], 0, 0
            extra_cost_b = 0
            backup_delay_cost = 0
            total_cost = cost_active + extra_cost_a
        placements.append({
            "Source": sfc.src.id,
            "Destination": sfc.dest.id,
            "SFC_ID": sfc.id,
            "Active_Path": path_active,
            "Active_Delay": delay_active,
            "Active_Cost": cost_active,
            "Backup_Path": path_backup,
            "Backup_Delay": delay_backup,
            "Backup_Cost": cost_backup,
            "Deploy_Cost_active": extra_cost_a,
            "Deploy_Cost_backup": extra_cost_b,
            "Backup_Delay_Cost": backup_delay_cost,
            "Total_Cost": total_cost
        })

    df = pd.DataFrame(placements)
    print("\nPlacement Results:")
    print(df.to_string(index=False))

    print("\nPhysical Node Resource Usage:")
    for node in physical_nodes:
        print(f"Node {node.id} - Used CPU: {node.used_cpu} / {node.cpu}")

    vmf_df = run_monitoring_placement(placements, Gp, physical_nodes)
    return df

if __name__ == "__main__":
    df_result = run_simulation()
    print("\nFinal Placement Summary:")
    print(df_result.to_string(index=False))
