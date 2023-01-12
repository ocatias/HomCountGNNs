import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Parameter
from torch_geometric.nn.inits import uniform

from Models.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    node_encoder = lambda x: x, edge_encoder = lambda x: x, use_node_encoder = True, graph_features = 0, num_mlp_layers = 1):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        print("Old GNN implementation.")

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.use_node_encoder = use_node_encoder
        self.graph_features = graph_features
        self.num_mlp_layers = num_mlp_layers
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)

        self.set_mlp(graph_features)

        ### Pooling function to generate whole-graph embeddings
        print(f"graph_pooling: {graph_pooling}")
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        # Attach graph level features
        if self.graph_features > 0:
            h_graph = torch.cat([h_graph,  batched_data.graph_features[:, 0:self.graph_features]], dim=1)

        h_graph = h_graph
        for layer in self.mlp:
            h_graph = layer(h_graph)

        if self.num_tasks == 1:
            h_graph = h_graph.view(-1, self.num_classes)
        else:
            h_graph.view(-1, self.num_tasks, self.num_classes)
        return h_graph

    def freeze_gnn(self, freeze = True):
        # Frezze GNN layers to allow us to tune the MLP separately
        for param in self.gnn_node.parameters():
            param.requires_grad = False

    def set_mlp(self, graph_features = 0, copy_emb_weights = False):
        self.graph_features = graph_features

        if self.num_mlp_layers == 1:

            # Create a new MLP, this is meant to be used if the number of graph level features changes
            new_linear_layer = Linear(self.emb_dim + graph_features, self.num_classes*self.num_tasks)

            if copy_emb_weights:
                new_linear_layer.weight.requires_grad = False
                new_linear_layer.weight[:, 0:self.emb_dim] = self.mlp[0].weight[:, 0:self.emb_dim].detach().clone()
                new_linear_layer.weight.requires_grad = True

                new_linear_layer.bias.requires_grad = False
                new_linear_layer.bias = Parameter(self.mlp[0].bias.detach().clone())
                new_linear_layer.bias.requires_grad = True

            self.mlp = ModuleList([new_linear_layer])
        else:
            hidden_size = self.emb_dim // 2
            self.mlp = ModuleList([Linear(self.emb_dim + graph_features, hidden_size), ReLU()])
            for _ in range(self.num_mlp_layers-2):
                self.mlp.append(Linear(hidden_size, hidden_size))
                self.mlp.append(ReLU())

            self.mlp.append(Linear(hidden_size, self.num_classes*self.num_tasks))


        






if __name__ == '__main__':
    GNN(num_tasks = 10)