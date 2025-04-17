from graphviz import Digraph

# Initialize the diagram
dot = Digraph(comment="AGI Problem Tree", format="png")
dot.attr(rankdir='LR')  # Left to right layout, can switch to 'TB' for top-bottom

# Helper to create unique node IDs
node_id_counter = [0]
def get_id():
    node_id_counter[0] += 1
    return f"node{node_id_counter[0]}"

# Recursive function to build the graph
def add_nodes(dot, tree, parent_id=None):
    for key, value in tree.items():
        current_id = get_id()
        dot.node(current_id, key, shape='box')
        if parent_id:
            dot.edge(parent_id, current_id)

        if isinstance(value, dict):
            add_nodes(dot, value, current_id)
        elif isinstance(value, list):
            for item in value:
                child_id = get_id()
                dot.node(child_id, item, shape='note')
                dot.edge(current_id, child_id)
        else:
            leaf_id = get_id()
            dot.node(leaf_id, str(value), shape='note')
            dot.edge(current_id, leaf_id)

# Data structure
agi_tree = {
    "AGI Problems": {
        "Human ability to validate models": {},
        "Too many models": {},
        "Too complex (smarter or more alien)": {},
        "Potential loss of control (self-improvement)": {
            "Alignment": {
                "Inner misalignment": {
                    "Deception": ["Evals", "Scalable oversight", "Red teaming"],
                    "Sycophancy": ["Scalable oversight", "Mechanistic interpretability"]
                },
                "Outer misalignment": {
                    "Reward hacking": ["Mechanistic interpretability", "Safety by design"],
                    "Goal misgeneralization": ["Goal preservation under self-modification", "Corrigibility"],
                    "Power-seeking behavior": ["Agent foundations", "Red teaming"]
                }
            },
            "Agent behavior monitoring (not CRO)": {},
            "Security/Governance": {
                "Client data and model weight security": {},
                "Avoiding dangerous model races": {},
                "Misuse (open-source harm)": {}
            },
            "Long-term existential risks": {
                "AI self-improvement": ["Turn it off", "Safety by design", "Corrigibility"]
            }
        }
    }
}

# Build the graph
add_nodes(dot, agi_tree)

# Render and save to file
dot.render("agi_problem_tree", view=True)
