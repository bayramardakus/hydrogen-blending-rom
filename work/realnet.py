"""realnet.py — Case Study 3 real SciGRID_gas German H-gas sub-network.

Builds a Network-compatible object directly from scigrid_subnet.json (22 nodes,
23 pipes, 716 km, two loops, extracted reproducibly by extract_subnet.py) and
reuses the generic build_rlc / build_reference / steady_state from network.py.

Demands are assigned deterministically (no hand-tuning): every non-supply node
withdraws a base load scaled by its adjacent pipe capacity, normalised so the
supply-trunk velocity is ~4 m/s — a realistic H-gas transmission operating point.
"""
import json, os, numpy as np
from gas_properties import mixture_properties, darcy_friction
import network as netmod
from network import steady_state, build_rlc, build_reference

SUBNET = os.path.join(os.path.dirname(__file__), 'scigrid_subnet.json')

class RealNetwork:
    def __init__(self, p_source_bar=80.0, demand_scale=None):
        d = json.load(open(SUBNET))
        self.meta = d
        self.n_nodes = d['n_nodes']
        self.source = d['supply']
        self.western_hub = d['western_hub']
        self.p_source_nom = p_source_bar * 1e5
        # pipes: (from, to, length[m], diameter[m])
        self.pipes = [(e['a'], e['b'], e['length_km']*1000.0, e['dia_m'])
                      for e in d['edges']]
        self.n_pipes = len(self.pipes)
        self.inc = np.zeros((self.n_nodes, self.n_pipes))
        for k,(a,b,L,D) in enumerate(self.pipes):
            self.inc[a,k]=+1.0; self.inc[b,k]=-1.0
        self.area = np.array([np.pi*D**2/4 for (_,_,_,D) in self.pipes])
        self.length = np.array([L for (_,_,L,_) in self.pipes])
        self.diam = np.array([D for (_,_,_,D) in self.pipes])
        self.total_km = self.length.sum()/1000.0
        self.coords = np.array([[nd['lon'], nd['lat']] for nd in
                                sorted(d['nodes'], key=lambda x:x['i'])])
        self.demand_nom = self._assign_demand(demand_scale)

    def _assign_demand(self, scale):
        # adjacent pipe cross-section capacity per node (deterministic)
        cap = np.zeros(self.n_nodes)
        for k,(a,b,L,D) in enumerate(self.pipes):
            cap[a]+=self.area[k]; cap[b]+=self.area[k]
        w = cap.copy(); w[self.source]=0.0
        w = w/w.sum()
        if scale is None: scale = 330.0   # total kg/s throughput (v~4 m/s trunk)
        dem = w*scale
        dem[self.source]=0.0
        return dem

def far_nodes(net, k=2):
    """The k delivery nodes furthest from the supply along the pipe graph.

    Scenario nodes are selected by this RULE rather than by hard-coded index.
    Node indices are an output of the extraction and, although now
    deterministic, they are not a physical property of the network; defining a
    disturbance by index would silently change the experiment if the extraction
    target or the dataset version changed.
    """
    import heapq
    n = net.n_nodes
    adj = {i: [] for i in range(n)}
    for k_, (a, b, L, D) in enumerate(net.pipes):
        adj[a].append((b, L)); adj[b].append((a, L))
    dist = {net.source: 0.0}
    pq = [(0.0, net.source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, np.inf):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    order = sorted((x for x in range(n) if x != net.source),
                   key=lambda x: (-dist.get(x, 0.0), x))
    return order[:k], {x: dist.get(x, 0.0) / 1000.0 for x in order[:k]}


def major_offtakes(net, k=2):
    """The k largest non-supply withdrawals, in descending order of nominal
    demand (ties broken by index).

    Disturbance scenarios are anchored here rather than at hard-coded indices.
    A contingency surge in a transmission system occurs at a large offtake - an
    industrial customer, a power station, a downstream distribution feed - so
    this is both the physically representative choice and a rule that survives a
    change of dataset version or extraction target.
    """
    d = net.demand_nom.copy()
    d[net.source] = -np.inf
    return sorted(range(net.n_nodes), key=lambda i: (-d[i], i))[:k]


def weakest_nodes(net, P0, k=2):
    """The k delivery nodes with the LOWEST nominal pressure.

    A pressure-security study must be stressed where the network is weakest:
    these are the nodes whose margin to the delivery floor is smallest, so a
    contingency there is the binding case an operator plans against. Selecting
    them by rule keeps the scenario reproducible and independent of the node
    numbering produced by the extraction.
    """
    cand = [i for i in range(net.n_nodes) if i != net.source]
    return sorted(cand, key=lambda i: (P0[i], i))[:k]


def steady_continuation(net, props, steps=8):
    """Solve steady state by ramping demand from 15% to 100% with warm starts,
    so the Newton solve converges at realistic (higher) throughput."""
    nN, nP = net.n_nodes, net.n_pipes
    x = np.concatenate([np.full(nN-1, net.p_source_nom/1e5 - 1.0), np.full(nP, 20.0)])
    P0 = Q0 = None
    for f in np.linspace(0.15, 1.0, steps):
        P0, Q0 = steady_state(net, props, demand=net.demand_nom*f, x0=x)
        x = np.concatenate([P0[[n for n in range(nN) if n!=net.source]]/1e5, Q0])
    return P0, Q0

def build(p_source_bar=72.0, demand_scale=500.0, y_nom=0.05):
    net = RealNetwork(p_source_bar, demand_scale)
    props = mixture_properties(y_nom)
    P0,Q0 = steady_state(net, props)
    return net, props, P0, Q0

if __name__ == "__main__":
    net, props, P0, Q0 = build()
    Pb = P0/1e5
    nonsrc = [n for n in range(net.n_nodes) if n!=net.source]
    print(f"RealNetwork: {net.n_nodes} nodes, {net.n_pipes} pipes, "
          f"{net.total_km:.0f} km, {net.n_pipes-net.n_nodes+1} loops")
    print(f"supply idx{net.source} @ {Pb[net.source]:.1f} bar | western hub idx{net.western_hub}")
    print(f"delivery pressures [bar]: min {Pb[nonsrc].min():.1f}  "
          f"max {Pb[nonsrc].max():.1f}  mean {Pb[nonsrc].mean():.1f}")
    v = np.abs(Q0)/(P0.mean()/props['a2']*net.area)
    print(f"pipe velocity [m/s]: min {v.min():.2f} max {v.max():.2f} (trunk {v[0]:.2f})")
    print(f"total demand {net.demand_nom.sum():.0f} kg/s | supply flow {Q0[0]:.0f} kg/s")
    print(f"mass-balance residual max {np.abs((net.inc@Q0+net.demand_nom)[nonsrc]).max():.2e}")
    A,B,C,info = build_rlc(net, props, P0, Q0)
    print(f"RLC state dimension: {info['ns']} states "
          f"({net.n_nodes-1} node pressures + {net.n_pipes} pipe flows)")
