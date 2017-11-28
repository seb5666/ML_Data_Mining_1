import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.cluster.hierarchy import dendrogram
import copy

# Create a random graph
def generate_graph(N = 20, num_clusters = 4, p_in = 0.99, p_out = 0.05):
    A = np.eye(N, dtype=int)

    cluster_assignements = np.repeat(np.arange(num_clusters), int(N/num_clusters))

    # create random edges
    for i in range(N):
        for j in range(i, N):
            p = p_in if cluster_assignements[i] == cluster_assignements[j] else p_out
            A[i,j] = max(A[i,j], p > np.random.rand())
            A[j,i] = A[i,j]
    return A, cluster_assignements

def generate_example_graph():
    # Graph example from the paper
    N = 16
    num_clusters = 2
    A = np.eye(N, dtype=int)
    cluster_assignements = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
    edges = [
             (0,4),
             (0,6),
             (1,2),
             (1,3),
             (1,4),
             (1,8),
             (2,3),
             (2,4),
             (2,5),
             (2,7),
             (3,6),
             (4,5),
             (5,6),
             (5,9),
             (6,7),
             (7,9),
             (8,9),
             (8,10),
             (8,11),
             (8,12),
             (8,14),
             (9,10),
             (9,14),
             (11,13),
             (12,13),
             (12,14),
             (13,14),
             (14,15)
             ]
    for (i,j) in edges:
        A[i,j] = 1
        A[j,i] = 1
    return A, cluster_assignements

#draw the graph
def draw_graph(A, clusters, num_clusters=4):
    N = A.shape[0]
    plt.axis('off')
    colors = ['r', 'g', 'b', 'y', 'b']
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    for i in range(num_clusters):
        nx.draw_networkx_nodes(G, pos, nodelist=clusters[i], node_color=colors[i % len(colors)], node_size=500, alpha=1)
    nx.draw_networkx_edges(G, pos, width=0.5)
    nx.draw_networkx_labels(G, pos, {i:i for i in range(N)})

def computeM(A, N):
    return (np.sum(A) + N) / 2

def cluster(A, t=3):
    N = A.shape[0]
    # Compute helper matrices
    degrees = np.sum(A, axis=1)
    D_inv = np.diag(1/degrees)
    D_inv_sqrt = np.diag(1/np.sqrt(degrees))

    #compute the number of edges
    M = computeM(A, N)

    def distance(P_i, P_j):
        return np.sqrt(np.sum((np.dot(D_inv_sqrt, P_i) - np.dot(D_inv_sqrt, P_j)) ** 2))

    def Δσ(l1, P_1, l2, P_2):
        return 1 / N * l1 * l2 / (l1 + l2) * (distance(P_1,P_2) ** 2)

    def find_to_merge(clusters):
        # find minimum Δσ for two clusters
        min_Δσ = None
        to_merge = None
        for i, cluster in clusters.items():
            for j, dist in cluster['neighbours'].items():
                if min_Δσ == None or dist < min_Δσ:
                    min_Δσ = dist
                    to_merge = (i, j)
        return (to_merge[0], to_merge[1], min_Δσ)

    # Given clusters, merge C1 and C2 into a new cluster with id new_id
    def merge_clusters(clusters, i, j, new_id):
        C1 = clusters[i]
        C2 = clusters[j]
        C3 = {}
        
        l1 = len(C1['nodes'])
        l2 = len(C2['nodes'])
        # Create new Cluster, merging C1 and C2
        C3['nodes'] = C1['nodes'].union(C2['nodes'])
        
        #compute the P_t for the new cluster
        C3['P_t'] = (l1 * C1['P_t'] + l2 * C2['P_t']) / (l1 + l2)

        new_neighbours = set(C1['neighbours'].keys()).union(set(C2['neighbours'].keys()))
        new_neighbours.remove(i)
        new_neighbours.remove(j)
        
        # Compute Δσ's for the new cluster
        C3['neighbours'] = {}
        for n in new_neighbours:
            C = clusters[n]
            l3 = len(C['nodes'])
            if n in C1['neighbours'].keys() and n in C2['neighbours'].keys():
                #apply Theorem 4
                x_A = (l1 + l3) * C1['neighbours'][n]
                x_B = (l2 + l3) * C2['neighbours'][n]
                x_C = (l3) * C1['neighbours'][j]
                new_Δσ = (x_A + x_B - x_C) / (l1 + l2 + l3)
            else:
                #apply Theorem 3
                new_Δσ = Δσ(l1+l2, C3['P_t'], l3, C['P_t'])
            
            C3['neighbours'][n] = new_Δσ
            C['neighbours'].pop(i, None)
            C['neighbours'].pop(j, None)
            C['neighbours'][new_id] = new_Δσ
        
        clusters[new_id] = C3
        clusters.pop(i)
        clusters.pop(j)

    def create_partition(clusters):
        return [C['nodes'] for C in clusters.values()]

    # Create transition matrix
    P = np.dot(D_inv, A)

    #Compute probability vectors
    P_t = np.eye(N)
    for i in range(t):
        P_t = np.dot(P, P_t)

    #create initial clusters
    A2 = A - np.eye(N)
    clusters = {}
    for i in range(N):
        neighbours = {}
        for j in range(N):
            if i != j and A[i,j] :
                neighbours[j] = Δσ(1, P_t[i], 1, P_t[j])
        clusters[i] = {
            'nodes': {i},
            'P_t': P_t[i],
            'neighbours': neighbours
        }
    # merge clusters repeatedly
    new_id = N
    build_tree = []
    Δσs = []
    partitions = [create_partition(clusters)]
    cum_dist = 0

    while(len(clusters) > 1):
        # find clusters to merge
        (i,j, min_Δσ) = find_to_merge(clusters)
        # Compute new partion
        merge_clusters(clusters, i, j, new_id)
        
        # For dendogram
        build_tree.append((new_id, i, j, cum_dist))
        cum_dist += min_Δσ
        
        # Keep track of partitions
        partitions.append(create_partition(clusters))
        
        # For evaluation of partitions
        Δσs.append(min_Δσ)
        
        new_id += 1

    return build_tree, partitions, Δσs


# Compute the modularity of a given partition
# Compute the modularity of a given partition
def modularity(partition, M):
    def edge_fraction(C1, C2):
        sum = 0
        # need to check if C1==C2, as we would be counting edges multiple times
        if (C1 == C2):
            nodes1 = list(C1)
            nodes2 = list(C2)
            for i in range(len(C1)):
                for j in range(i, len(C2)):
                    if (A[nodes1[i],nodes2[j]] == 1):
                        sum += 1
        else:
            for i in C1:
                for j in C2:
                    if (A[i,j] == 1):
                        sum += 1
        return sum / M
    
    def edges_bound(C1):
        sum = 0
        for C in partition:
            e = edge_fraction(C1, C)
            sum += e
        return sum
    
    modularity = 0.0
    for C in partition:
        e_C = edge_fraction(C,C)
        a_C = edges_bound(C)
        modularity += e_C - (a_C**2)
    return modularity


def increase_ratios(Δσs):
    η = []
    for i in range(0, len(Δσs) - 1):
        if Δσs[i] == 0:
            η.append(0)
        else: 
            η.append(Δσs[i+1]/Δσs[i])
    return  np.flip(η, axis=0)


def plot_dendogram(build_tree):
    plt.figure()
    #create linkage matrix
    Z = np.array([[id1, id2, dist, 0] for (j, (new_id, id1, id2, dist)) in enumerate(build_tree)], dtype=np.double)
    dendrogram(Z)
    # plt.savefig("plots/karate_dendogram")
    plt.figure()
    Z = np.array([[id1, id2, j+1, 0] for (j, (new_id, id1, id2, dist)) in enumerate(build_tree)], dtype=np.double)
    dendrogram(Z)

def plot_eval(N, Qs, η):
    plt.figure()
    plt.title(r'Increase ratios $\eta_k$')
    plt.plot(range(2, len(η) + 2),η)
    plt.xlabel("Number of partitions k")
    # plt.savefig("plots/football_graph_eta")
    plt.figure()
    plt.title("Modularity Q")
    plt.plot(range(1,len(Qs) + 1), Qs)
    plt.xlabel("Number of partitions k")
    # plt.savefig("plots/football_graph_Q")

def rand_index(P1, P2):
    N = sum([len(C) for C in P1])
    
    A = sum([len(C) ** 2 for C in P1])
    B = sum([len(C) ** 2 for C in P2])

    C = 0
    for C1 in P1:
        for C2 in P2:
            C += len(set.intersection(C1, C2)) ** 2

    return ((N**2) * C - A * B) / (1/2 * (N **2) * (A + B) - A * B )

times = []
for N in [20, 100, 500, 1000]:
    print(N)
    num_clusters = 10
    A, cluster_assignements = generate_graph(N = N, num_clusters = num_clusters, p_in = 0.95, p_out = 0.1)
#    A, cluster_assignements = generate_example_graph()
#    N = A.shape[0]
    cluster_nodes = [list(np.where(cluster_assignements == i)[0]) for i in range(num_clusters)]
    plt.figure()
    M = computeM(A, N)
    print("M: {}".format(M))
    
    start_time = time.time()
    build_tree, partitions, Δσs = cluster(A)
    print("--- %s seconds ---" % (time.time() - start_time))

    Qs = [modularity(partitions[N-i], M) for i in range(1, 20)]
    η = increase_ratios(Δσs) + 2

    max_K_Q = np.argmax(Qs[:20]) + 1
    max_K_η = np.argmax(η[:20]) + 2
    print("Best k according to Q: {}".format(max_K_Q))
    print("Best k according to η: {}".format(max_K_η))
    print("Rand index: {}".format(rand_index(partitions[N - max_K_Q], [set(C) for C in cluster_nodes])))
    print("Rand index with correct number of clusters: {}".format(rand_index(partitions[N - num_clusters], [set(C) for C in cluster_nodes])))
    if N < 200:
        draw_graph(A, cluster_nodes, num_clusters)
        plot_dendogram(build_tree)
        plot_eval(N, Qs, η)
        plt.show()

print("Done")

#
## In[213]:
#
#
#best_num_clusters = np.argmax(η[2:15]) + 2
#plt.title("Best increase ratio = {}".format(best_num_clusters))
#draw_graph(partitions[N-best_num_clusters], best_num_clusters)
## plt.savefig("plots/karate_graph_partition2")
## plt.savefig(filename="test.png", format="png")
#plt.figure()
#best_num_clusters_Qs = np.argmax(Qs[1:]) + 2
#plt.title("Best modularity Q = {}".format(best_num_clusters_Qs))
#draw_graph(partitions[N-best_num_clusters_Qs], best_num_clusters_Qs)
## plt.savefig("plots/karate_graph_partition4")

