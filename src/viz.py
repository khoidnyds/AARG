import plotly.graph_objects as go

import networkx as nx


class Viz():
    def __init__(self, path, out_dir):
        self.path_graph = path
        self.html = out_dir.joinpath("graph.html")

    def generate(self):
        # import the graph
        G = nx.read_gpickle(self.path_graph)
        pos = nx.spring_layout(G)

        # generate edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(x=edge_x,
                                y=edge_y,
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='name',
                                mode='lines')

        # generate node traces
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x,
                                y=node_y,
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(
                                    showscale=True,
                                    # colorscale options
                                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                                    colorscale='Reds',
                                    reversescale=False,
                                    color=[],
                                    size=10,
                                    colorbar=dict(
                                        thickness=15,
                                        title='Node Connections',
                                        xanchor='left',
                                        titleside='right'
                                    ),
                                    line_width=2))

        # add node color and degree
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'# of connections: {str(len(adjacencies[1]))}')

        # generate the plot
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            width=1200,
                            height=800,
                            title='<br>Protein protein network of ARGs in STRING',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.write_html(self.html)
