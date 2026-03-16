import os
import webbrowser

import graphviz
import networkx as nx
from pyvis.network import Network


class TreeVisualizer:
    def __init__(self, tree, level, node=None):
        if node is None:
            self.root = tree.root
        else:
            self.root = node
        self.graph = nx.DiGraph()
        self.node_styles = {}
        self.node_shapes = {}
        self.node_term = {}
        self.node_colors = {}

        self.adversarial = tree.adversarial

        # build the graph
        self.build_graph(level, node=self.root)

    def build_graph(self, level, node, parent_id=None):
        node_id = node.id  # Unique identifier
        self.graph.add_node(node_id, label=str(node_id), available_actions=node.available_actions)

        # Determine node style
        if node.is_chance:
            self.node_styles[node_id] = "dashed"
        else:
            self.node_styles[node_id] = "solid"

        if node.is_terminal:
            self.node_shapes[node_id] = "square"
            self.node_term[node_id] = True
        else:
            self.node_shapes[node_id] = "circle"
            self.node_term[node_id] = False

        if self.adversarial:
            if node.color == 1:
                self.node_colors[node_id] = "white", "black"
            elif node.color == 2:
                self.node_colors[node_id] = "black", "white"
            else:
                raise ValueError("Invalid color")
        else:
            self.node_colors[node_id] = "white", "black"

        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id, label=str(node._action))

        for action, child in node.children.items():
            if child is not None and level > 0:
                self.build_graph(level-1, child, node_id)

    def visualize(self, filename="tree_visualization/tree_visualization"):
        dot = graphviz.Digraph(format="png")

        for node_id, data in self.graph.nodes(data=True):
            label = data.get("label", "")
            style = self.node_styles.get(node_id, "solid")
            dot.node(str(node_id),
                     label,
                     shape=self.node_shapes[node_id],
                     style="filled," + style,
                     fillcolor=self.node_colors[node_id][0],
                     fontcolor=self.node_colors[node_id][1],
                     )

        for u, v, data in self.graph.edges(data=True):
            action_label = data.get("label", "")
            dot.edge(str(u), str(v), label=action_label)

        # Add dangling edges for available actions
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            if node and len(node['available_actions']) > 0 and not self.node_term[node_id]:
                for action in node['available_actions']:
                    dummy_node_id = f"dummy_{node_id}_{action}"
                    dot.node(dummy_node_id, label="", shape="point", width="0.01")
                    dot.edge(str(node_id), dummy_node_id, label=str(action), style="dotted")

        dot.render(filename, view=True)


class InteractiveTreeVisualizer:
    def __init__(self, tree, level, node=None):
        self.root = tree.root if node is None else node
        self.graph = nx.DiGraph()
        self.node_shapes = {}
        self.node_colors = {}
        self.adversarial = tree.adversarial

        self.build_graph(level, self.root)

    def build_graph(self, level, node, parent_id=None):
        node_id = node.id

        self.graph.add_node(
            node_id,
            label=str(node_id),
            available_actions=node.available_actions,
            is_terminal=node.is_terminal,
            is_chance=node.is_chance,
            color=node.color if self.adversarial else None,
            game_reward=getattr(node, "game_reward", None),
            value=getattr(node, "value", None),
            visits=getattr(node, "visits", None),
            score=getattr(node, "score", None),
            q_vector_sum=(
                node.q_vector.sum()
                if hasattr(node, "q_vector") and node.q_vector is not None
                else None
            ),
            is_root=getattr(node, "is_root", False),
        )

        # Node shape
        self.node_shapes[node_id] = "box" if node.is_terminal else "circle"

        # Node color
        if self.adversarial:
            if node.color == 1:
                self.node_colors[node_id] = ("white", "black")
            elif node.color == 2:
                self.node_colors[node_id] = ("black", "white")
            else:
                raise ValueError("Invalid color for adversarial")
        else:
            self.node_colors[node_id] = ("white", "black")

        # Edge from parent
        if parent_id is not None:
            self.graph.add_edge(
                parent_id, node_id, label=str(getattr(node, "_action", ""))
            )

        # Recurse
        if level > 0:
            for _, child in node.children.items():
                if child is not None:
                    self.build_graph(level - 1, child, node_id)

    def visualize(self, filename="tree_visualization/mcts_tree.html"):
        net = Network(height="900px", width="100%", directed=True)

        net.set_options(
            """
        {
          "nodes": {
            "borderWidth": 2,
            "font": {
              "size": 14,
              "face": "Arial",
              "align": "center",
              "vadjust": 0
            }
          },
          "edges": {
            "arrows": "to",
            "color": {
              "color": "#000000",
              "highlight": "#000000",
              "hover": "#000000"
            }
          },
          "layout": {
            "hierarchical": {
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 120,
              "nodeSpacing": 60
            }
          },
          "physics": { "enabled": false },
          "interaction": { "hover": false }
        }
        """
        )

        # -----------------
        # Add real nodes
        # -----------------
        for node_id, data in self.graph.nodes(data=True):
            fill, border = self.node_colors[node_id]
            shape = self.node_shapes[node_id]

            info_text = (
                f"game_reward:\t{data.get('game_reward')}\n"
                f"score:\t{data.get('score')}\n"
                f"visits:\t{data.get('visits')}\n"
                f"value:\t{data.get('value')}\n"
                f"Q-vec:\t{data.get('q_vector_sum')}"
            )

            if shape == "box":
                net.add_node(
                    node_id,
                    label=str(node_id),
                    shape="box",
                    title="",
                    game_info=info_text,
                    color={"background": fill, "border": border},
                    font={"size": 14, "face": "Arial", "align": "center"},
                    widthConstraint={"minimum": 30},
                    heightConstraint={"minimum": 30},
                )
            else:
                net.add_node(
                    node_id,
                    label=str(node_id),
                    shape="circle",
                    size=30,
                    title="",
                    game_info=info_text,
                    color={"background": fill, "border": border},
                    font={"size": 14, "face": "Arial", "align": "center"},
                )

        # -----------------
        # Add real edges
        # -----------------
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(
                u,
                v,
                label=data.get("label"),
                color="black",
                arrows="to",
            )

        # -----------------
        # Dummy nodes + dangling edges
        # -----------------
        for node_id, data in self.graph.nodes(data=True):
            actions = data.get("available_actions", [])
            if actions and not data.get("is_terminal"):
                for action in actions:
                    dummy_id = f"dummy_{node_id}_{action}"

                    net.add_node(
                        dummy_id,
                        shape="circle",
                        # size=0,
                        physics=False,
                        label="",
                        title="",
                        color={
                            "background": "rgba(0,0,0,0)",
                            "border": "rgba(0,0,0,0)",
                        },
                        font={"size": 0, "color": "rgba(0,0,0,0)"},
                        widthConstraint={"minimum": 30},
                        heightConstraint={"minimum": 30},

                    )

                    net.add_edge(
                        node_id,
                        dummy_id,
                        label=str(action),
                        dashes=True,
                        arrows="to",
                        color="black",
                    )

        # -----------------
        # Write HTML
        # -----------------
        net.write_html(filename)

        # -----------------
        # Inject JS fixes
        # -----------------
        try:
            with open(filename, "r", encoding="utf-8") as f:
                html = f.read()

            inject = """
<script type="text/javascript">
document.addEventListener("DOMContentLoaded", function () {
  if (typeof network === "undefined") return;

  // Disable interaction for dummy nodes
  for (const nodeId in network.body.nodes) {
    if (String(nodeId).startsWith("dummy_")) {
      const node = network.body.nodes[nodeId];
      if (node) {
        node.options.interaction = false;
        node.options.selectable = false;
        node.options.hover = false;
      }
    }
  }

  // Disable interaction for edges connected to dummy nodes
  for (const edgeId in network.body.edges) {
    const edge = network.body.edges[edgeId];
    if (
      String(edge.fromId).startsWith("dummy_") ||
      String(edge.toId).startsWith("dummy_")
    ) {
      edge.options.interaction = false;
      edge.options.hover = false;
    }
  }

  // Click popup for real nodes
  network.on("click", function (params) {
    if (params.nodes && params.nodes.length) {
      const nodeId = params.nodes[0];
      if (String(nodeId).startsWith("dummy_")) return;
      const node = network.body.data.nodes.get(nodeId);
      if (!node) return;
      const info = node.game_info || "";
      if (info) alert(info);
    }
  });
});
</script>
</body>
"""
            if "</body>" in html:
                html = html.replace("</body>", inject)
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html)

        except Exception as e:
            print("HTML injection failed:", e)

        # -----------------
        # Open browser
        # -----------------
        try:
            webbrowser.open("file://" + os.path.realpath(filename))
        except Exception:
            print(f"Visualization saved to: {filename}")
