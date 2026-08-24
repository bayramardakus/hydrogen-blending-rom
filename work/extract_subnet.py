"""Deterministic, reproducible extraction of a German H-gas transmission
sub-network from the SciGRID_gas IGGIELGN dataset (v1.1.2).

Produces scigrid_subnet.json with 22 nodes / 23 pipes / two loops, spanning
an eastern gas supply and a western hub. Fully reproducible from the named
dataset (no hand-picking): the algorithm is (1) largest DE H-gas connected
component, (2) eastern + western highest-degree hubs, (3) shortest east-west
corridor, (4) breadth-first accretion of nearest nodes until |V|=22, (5) reduce
to |E|=23 by keeping a min-total-length spanning tree plus the two longest
chords -> exactly two independent loops.
"""
import pandas as pd, ast, json, networkx as nx, numpy as np, os

DATA = os.path.join(os.path.dirname(__file__), '..', 'scigrid', 'data')
N_TARGET = 22
N_LOOPS = 2

def _lit(x):
    try: return ast.literal_eval(x)
    except Exception: return None

def build_full_graph():
    ps = pd.read_csv(os.path.join(DATA,'IGGIELGN_PipeSegments.csv'), sep=';')
    ps['ccs']=ps['country_code'].apply(lambda x: _lit(x) or [])
    ps['nodes']=ps['node_id'].apply(_lit)
    ps['par']=ps['param'].apply(_lit)
    de = ps[ps['ccs'].apply(lambda l: len(l)>0 and all(c=='DE' for c in l))].copy()
    de['isH']=de['par'].apply(lambda p: p.get('is_H_gas',0))
    G=nx.Graph()
    for _,r in de[de['isH']==1].iterrows():
        nn=r['nodes']; p=r['par']
        if nn and len(nn)==2 and nn[0]!=nn[1]:
            ln=float(p.get('length_km',0.0)); dia=float(p.get('diameter_mm',0.0))
            pr=float(p.get('max_pressure_bar',84.0))
            if G.has_edge(*nn):  # keep the larger-diameter parallel line
                if dia<=G[nn[0]][nn[1]]['dia']: continue
            G.add_edge(nn[0],nn[1], length=ln, dia=dia, pmax=pr)
    Gc=G.subgraph(max(nx.connected_components(G), key=len)).copy()
    nd=pd.read_csv(os.path.join(DATA,'IGGIELGN_Nodes.csv'), sep=';')
    coord={r['id']:(float(r['lat']),float(r['long'])) for _,r in nd.iterrows()}
    return Gc, coord

def extract(Gc, coord):
    # eastern & western highest-degree hubs (tie-break by degree then longitude)
    deg=dict(Gc.degree())
    east=max(Gc.nodes, key=lambda n:(deg[n], coord[n][1]))     # high deg, large long
    west=max(Gc.nodes, key=lambda n:(deg[n],-coord[n][1]))     # high deg, small long
    # shortest east-west corridor by pipe length
    corridor=nx.shortest_path(Gc, east, west, weight='length')
    chosen=list(dict.fromkeys(corridor))
    # accrete neighbours until |V| = N_TARGET, preferring nodes that close loops
    # (link to >=2 already-chosen nodes) so the induced subgraph carries >=2 chords
    while len(chosen)<N_TARGET:
        cset=set(chosen); cand={}
        for u in chosen:
            for v in Gc.neighbors(u):
                if v in cset: continue
                links=sum(1 for w in Gc.neighbors(v) if w in cset)
                minl=min(Gc[v][w]['length'] for w in Gc.neighbors(v) if w in cset)
                # prefer more links into chosen (loop-closing), then shorter pipe
                cand[v]=(-links, minl, v)
        if not cand: break
        nxt=min(cand, key=lambda v:cand[v])
        chosen.append(nxt)
    H=Gc.subgraph(chosen).copy()
    # reduce to exactly N_TARGET-1 + N_LOOPS edges: min-length spanning tree + longest chords
    T=nx.minimum_spanning_tree(H, weight='length')
    chords=sorted([(H[u][v]['length'],u,v) for u,v in H.edges if not T.has_edge(u,v)],
                  reverse=True)
    S=nx.Graph(); S.add_nodes_from(H.nodes(data=True))
    for u,v in T.edges: S.add_edge(u,v,**H[u][v])
    for l,u,v in chords[:N_LOOPS]: S.add_edge(u,v,**H[u][v])
    return S, east, west

def main():
    Gc, coord = build_full_graph()
    S, _e, _w = extract(Gc, coord)
    nodes=list(S.nodes())
    deg=dict(S.degree())
    # supply = eastern highest-degree node
    east=max(nodes, key=lambda n:(deg[n], coord[n][1]))
    # western hub = the gateway that gates the western/northern branches: highest
    # length-weighted betweenness among nodes west of the supply (an articulation
    # carrying every western/northern delivery has high betweenness), interior deg>=2
    btw=nx.betweenness_centrality(S, weight='length')
    wcand=[n for n in nodes if coord[n][1]<coord[east][1] and deg[n]>=2 and n!=east]
    west=max(wcand, key=lambda n:(btw[n], -coord[n][1]))
    # DETERMINISTIC INDEX ASSIGNMENT.
    # The node ordering previously came from NetworkX's unordered node view, so
    # a re-run of this script reproduced the same TOPOLOGY but relabelled the
    # nodes: 18 of 22 indices changed, and with them every index-based statement
    # in the paper and every index-based scenario definition in the case-study
    # scripts. Sorting by SciGRID_gas identifier makes the labelling itself
    # reproducible, so that "supply = idx 0, western hub = idx k" is a stable,
    # citable fact rather than an artefact of one particular run.
    order=[east]+sorted(n for n in nodes if n!=east)
    idx={n:i for i,n in enumerate(order)}
    edges=[]
    for u,v in S.edges():
        # canonical orientation a < b: NetworkX yields (u,v) in an unspecified
        # order, and the orientation fixes the SIGN convention of the incidence
        # matrix, hence the sign of every reported pipe flow.
        if idx[u] > idx[v]:
            u, v = v, u
        edges.append(dict(a=idx[u], b=idx[v],
                          length_km=round(S[u][v]['length'],3),
                          dia_m=round(S[u][v]['dia']/1000.0,4),
                          pmax_bar=S[u][v]['pmax']))
    edges=sorted(edges, key=lambda e: (min(e['a'],e['b']), max(e['a'],e['b'])))
    out=dict(
        nodes=[dict(i=idx[n], scigrid_id=n, lat=coord[n][0], lon=coord[n][1],
                    degree=S.degree(n)) for n in order],
        edges=edges,
        supply=idx[east], western_hub=idx[west],
        n_nodes=len(order), n_pipes=len(edges),
        total_km=round(sum(e['length_km'] for e in edges),1),
        n_loops=len(edges)-len(order)+1,
        diameters_m=[round(min(e['dia_m'] for e in edges),3),
                     round(max(e['dia_m'] for e in edges),3)])
    with open(os.path.join(os.path.dirname(__file__),'scigrid_subnet.json'),'w') as f:
        json.dump(out,f,indent=1)
    print(f"nodes={out['n_nodes']} pipes={out['n_pipes']} loops={out['n_loops']} "
          f"total={out['total_km']} km  dia={out['diameters_m']} m")
    print(f"supply(east)=idx0 {east} lat{coord[east][0]:.2f} lon{coord[east][1]:.2f} deg{S.degree(east)}")
    print(f"western_hub=idx{idx[west]} {west} lat{coord[west][0]:.2f} lon{coord[west][1]:.2f} deg{S.degree(west)}")
    degs=sorted(S.degree, key=lambda t:t[1], reverse=True)[:4]
    print('top degrees in subnet:', [(idx[n],d) for n,d in degs])

if __name__=='__main__':
    main()
