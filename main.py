import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt


# Load Dataset
def load_graph(path):  
    G = nx.read_edgelist(path, delimiter=" ", create_using=nx.Graph(), nodetype=int)
    return G

# Friend Recommendation
def recommend_friends(graph, node, top_k=100): 
    neighbors = set(graph.neighbors(node))
    scores = {}
    for potential_friend in graph.nodes():
        if potential_friend == node or potential_friend in neighbors:
            continue
        common_neighbors = len(neighbors.intersection(set(graph.neighbors(potential_friend))))
        if common_neighbors > 0:
            scores[potential_friend] = common_neighbors
    
    recommendations = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return recommendations

# Visualize graph
def visualize_subgraph(graph, sampled_nodes):
    subgraph = graph.subgraph(sampled_nodes) 
    plt.figure(figsize=(10, 8))
    nx.draw(
        subgraph,
        with_labels=True,
        node_size=300,
        node_color='lightblue',
        font_size=6,
        font_color='black',
        edge_color='red'
    )
    plt.title("Visualizing the Graph")
    plt.show()

# Testing the Model
def test_model(graph):
    nodes = list(graph.nodes())
    ground_truth = {}
    recommendations = {}
    
    sampled_nodes = random.sample(nodes, 100)  # Sample x nodes
    for node in sampled_nodes:
        ground_truth[node] = random.sample(nodes, 3030) 
        recommendations[node] = recommend_friends(graph, node, top_k=100) 
    
    y_true = [1 if rec in ground_truth[node] else 0 for node in ground_truth for rec in recommendations[node]]
    y_pred = [1 for _ in y_true]  

    if not y_true or not y_pred:
        print("Empty recommendations or ground truth.")
        return [0], [0], sampled_nodes
    
    return y_true, y_pred, sampled_nodes

# Main
if __name__ == "__main__":
    datapath = r"E:\archive\ego-Facebook\facebook_combined.txt"
    graph = load_graph(datapath)
    
    # Test the model
    y_true, y_pred, sampled_nodes = test_model(graph)
    
    # Evaluate performance
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Visualize graph
    visualize_subgraph(graph, sampled_nodes)
    