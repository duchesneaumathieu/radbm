def maximal_prefix(x, y):
    k = 0
    for a, b in zip(x, y):
        if a == b:
            k += 1
        else: break
    return x[:k]

class Node(dict):
    def __init__(self, tag=False):
        self.tag = tag
        self.parent = None
        self.prefix = ()
        self.max_height = 0 #farthest leaf
        self.min_height = 0 #closest leaf
        self.size = 1 #number of nodes in sub-tree
    
    def __setitem__(self, k, v):
        if not isinstance(k, tuple):
            raise ValueError(f'key must be tuple, got {type(k)}')
        if not isinstance(v, Node):
            raise ValueError(f'value must be node, got {type(v)}')
        super().__setitem__(k, v)
        v.parent = self
        v.prefix = self.prefix + k
        
    def update_info(self):
        self.size = sum(child.size for child in self.values()) + 1
        self.max_height = max((child.max_height for child in self.values()), default=-1) + 1
        self.min_height = min((child.min_height for child in self.values()), default=-1) + 1
        
    def propagate_update(self):
        self.update_info()
        if self.parent is not None:
            self.parent.propagate_update()
        
    def __repr__(self):
        return str(self.prefix)

class Trie(object):
    def __init__(self,):
        self.root = Node()
        self._propagate_update(self.root)
        
    def _propagate_update(self, node):
        node.propagate_update()
        self.size = self.root.size
        self.max_height = self.root.max_height
        self.min_height = self.root.min_height
        return self
        
    def insert(self, document):
        document = tuple(document)
        suffix = document
        node = self.root
        while True:
            if not suffix:
                node.tag = True
                return self._propagate_update(node)
            prefixes = [maximal_prefix(suffix, branch) for branch in node]
            non_empty_prefix_count = sum(map(bool, prefixes))
            if non_empty_prefix_count == 0:
                node[suffix] = Node(tag=True)
            elif non_empty_prefix_count == 1:
                prefix, branch = [(p, b) for (p, b) in zip(prefixes, node) if p][0]
                if len(branch) <= len(prefix): # <= could be safely replaced with == since a prefix is alway smaller or equal
                    #following an existing branch
                    suffix = suffix[len(prefix):]
                    node = node[branch]
                elif len(suffix) <= len(prefix): # again, <= could be safely replaced with ==
                    #creating an intermediate node (ab->abcdef) ==> (ab->abcd->abcde)
                    node[suffix] = Node(tag=True)
                    node[suffix][branch[len(prefix):]] = node[branch]
                    del node[branch]
                    return self._propagate_update(node[suffix])
                else: # common prefix
                    #creating a spliting node (ab->abcde) ==> (ab->abc->(abcde, abced))
                    node[prefix] = Node()
                    node[prefix][suffix[len(prefix):]] = Node(tag=True)
                    node[prefix][branch[len(prefix):]] = node[branch]
                    del node[branch]
                    return self._propagate_update(node[prefix])
            else:
                raise RuntimeError(f'node={str(node)} contains branches with non-empty common prefixes.')
                
    def search(self, query):
        raise NotImplementedError()
        
    def __repr__(self):
        return f'size={self.size}, min_height={self.min_height}, max_height={self.max_height}'
                   
    def graph(self, max_depth=float('inf'), node_label=lambda node: str(node), branch_label=lambda branch: str(branch)):
        import graphviz
        G = graphviz.Graph()
        nodes = [(0, self.root)]
        while nodes:
            depth, node = nodes.pop()
            G.node(name=str(node.prefix), label=node_label(node), style='filled' if node.tag else None)
            if depth < max_depth:
                for k in sorted(node.keys()):
                    v = node[k]
                    nodes.append((depth+1, v))
                    G.edge(str(node.prefix), str(v.prefix), label=branch_label(k))
        return G