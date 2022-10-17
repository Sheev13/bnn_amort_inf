import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Represents the deterministic encoder for a conditional neural process"""
        
    def __init__(
        self,
        data_dim,
        hidden_dims,
        representation_dim,
        activation,
    ):
        super(Encoder, self).__init__()
        self.input_dim = data_dim * 2  # context and target for each dim
        self.hidden_dims = hidden_dims
        self.representation_dim = representation_dim
        self.activation = activation

        self.network = nn.ModuleList()

        self.network.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.network.append(self.activation)

        for i in range(1, len(hidden_dims)):
            self.network.append(
                nn.Linear(
                    self.hidden_dims[i - 1],
                    self.hidden_dims[i],
                )
            )
            self.network.append(self.activation)

        self.network.append(nn.Linear(self.hidden_dims[-1], self.representation_dim))
        self.network.append(nn.Identity())
        
    def forward(self, x_c, y_c):
        # z is shape (batch_size, 2 * data_dim)
        z = torch.cat((x_c, y_c), dim=1)
        assert z.shape[1] == self.input_dim
        
        for layer in self.network:
            z = layer(z)
            
        # z is shape (batch_size, representation_dim)
        assert len(z.shape) == 2
        assert z.shape[1] == self.representation_dim
        return z
    
    
class Decoder(nn.Module):
    """Represent the determinisitc decoder for a condition nerual process"""
    
    def __init__(
        self,
        representation_dim,
        hidden_dims,
        data_dim,
        activation,
        ):
        super(Decoder, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims
        self.output_dim = data_dim * 2  # mean and var for each dim
        self.data_dim = data_dim
        self.activation = activation
        
        self.network = nn.ModuleList()
        
        self.network.append(nn.Linear(self.representation_dim + self.data_dim, self.hidden_dims[0]))
        self.network.append(self.activation)
        
        for i in range(1, len(hidden_dims)):
            self.network.append(
                nn.Linear(
                    self.hidden_dims[i - 1],
                    self.hidden_dims[i],
                )
            )
            self.network.append(self.activation)
        
        self.network.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
    def forward(self, representation, x_t):
        # representation is shape (1, representation_dim)
        assert representation.shape[0] == 1
        assert representation.shape[1] == self.representation_dim
        num_test = x_t.shape[0]
        # representation is shape (num_test, representation_dim)
        representation = representation.repeat(num_test, 1)
        
        z = torch.cat((representation, x_t), dim=1)
        assert z.shape[1] == self.representation_dim + self.data_dim
        
        for layer in self.network:
            z = layer(z)
            
        assert z.shape[1] == self.output_dim
        
        pred_mu, pred_logstd = torch.split(
            z,
            self.data_dim,
            dim=1,
        )
        
        return pred_mu, pred_logstd
    
    
class GCNP(nn.Module):
    """Represents a vanilla Gaussian conditional neural process"""
    
    def __init__(
        self,
        data_dim,
        enc_hidden_dims,
        representation_dim,
        dec_hidden_dims,
        activation = nn.ReLU(),
    ):
        super(GCNP, self).__init__()
        self.data_dim = data_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.representation_dim = representation_dim
        self.dec_hidden_dims = dec_hidden_dims
        self.activation = activation
        
        self.encoder = Encoder(
            self.data_dim,
            self.enc_hidden_dims,
            self.representation_dim,
            self.activation,
        )
        
        self.decoder = Decoder(
            self.representation_dim,
            self.dec_hidden_dims,
            self.data_dim,
            self.activation,
        )
        
    def forward(self, x_c, y_c, x_t,):
        
        # average over batch dimension
        # representation is shape (1, representation_dim)
        representation = self.encoder(x_c, y_c).mean(dim=0).unsqueeze(0)
        return self.decoder(representation, x_t)
    
    def neg_ll(self, x_c, y_c, x_t, y_t):
        y_t_mu, y_t_logstd = self.forward(x_c, y_c, x_t)
        dist = torch.distributions.normal.Normal(y_t_mu, y_t_logstd.exp())
        return - dist.log_prob(y_t).sum().squeeze()