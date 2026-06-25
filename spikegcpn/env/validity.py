import torch


def is_dag(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """
    Return True if the directed graph has no cycles (i.e. is a DAG).

    Uses iterative DFS with three-color marking (white/gray/black).
    Gray nodes are on the current DFS stack; a back-edge to a gray node
    means a cycle exists.
    """
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes

    for start in range(num_nodes):
        if color[start] != WHITE:
            continue
        # Iterative DFS using an explicit stack of (node, iterator) pairs.
        stack = [(start, iter(adj[start]))]
        color[start] = GRAY
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if color[child] == GRAY:
                    return False  # back edge → cycle
                if color[child] == WHITE:
                    color[child] = GRAY
                    stack.append((child, iter(adj[child])))
            except StopIteration:
                color[node] = BLACK
                stack.pop()

    return True


def would_create_cycle(edge_index: torch.Tensor, num_nodes: int, src: int, dst: int) -> bool:
    """
    Return True if adding the directed edge src → dst would create a cycle
    in the current graph.

    Self-loops always create a cycle (src == dst).
    Otherwise we check whether dst can already reach src via BFS.
    """
    if src == dst:
        return True

    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        s_list = edge_index[0].tolist()
        d_list = edge_index[1].tolist()
        for s, d in zip(s_list, d_list):
            adj[s].append(d)

    # BFS from dst; if we reach src, adding src→dst would close a cycle.
    visited: set[int] = set()
    queue = [dst]
    while queue:
        u = queue.pop()
        if u == src:
            return True
        if u in visited:
            continue
        visited.add(u)
        queue.extend(adj[u])

    return False


def all_sinks_reachable(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_source: int,
    num_sink: int,
) -> bool:
    """
    Return True if every sink node is reachable from at least one source node.

    Source nodes are indices [0, num_source).
    Sink nodes are indices [num_nodes - num_sink, num_nodes).
    """
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)

    # BFS / DFS from all source nodes simultaneously.
    visited: set[int] = set(range(num_source))
    stack = list(range(num_source))
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)

    sink_start = num_nodes - num_sink
    return all(i in visited for i in range(sink_start, num_nodes))
