"""
Neural network models for PhyCRNet.
Includes PhyCRNet and its components (ConvLSTM, ResBlock).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhyCRNet(nn.Module):
    """Physics-informed Convolutional-Recurrent Network."""
    
    def __init__(self, ch=4, hidden=192, upscale=1, dropout_rate=0.2):
        super().__init__()
        
        self.input_ch = ch
        self.output_ch = ch
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ConvLSTM layers
        self.conv_lstm = DeepConvLSTM(hidden, hidden, num_layers=3, kernel_size=5, padding=2)
        
        # Residual block
        self.residual_block = ResidualBlock(hidden, hidden)
        
        # Decoder for main fields (U, V, T, P)
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate/2),
            nn.Conv2d(64, ch*(upscale**2), 3, padding=1),
            nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()
        )
        
        
        self._initialize_weights()
        
        
        self.up = upscale

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor [B×C×H×W]
            
        Returns:
            torch.Tensor: Output tensor [B×C×H×W] (U, V, T, P)
        """
        # Encoding
        z = self.enc(x)                           # B×hidden×H×W
        
        # ConvLSTM processing
        z = z.unsqueeze(1)                        # B×1×hidden×H×W
        z, _ = self.conv_lstm(z)                  # B×1×hidden×H×W
        z = z.squeeze(1)                          # B×hidden×H×W
        
        # Residual processing
        z = self.residual_block(z)                # B×hidden×H×W
        
        # Decoding main fields
        output = self.dec(z)                      # B×C×H×W (U, V, T, P)
        
        return output

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class DeepConvLSTM(nn.Module):
    """Multi-layer ConvLSTM with attention mechanism."""
    
    def __init__(self, in_channels, hidden_channels, num_layers=3, kernel_size=5, padding=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # ConvLSTM layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(ConvLSTM(in_channels, hidden_channels, kernel_size, padding))
            else:
                self.cells.append(ConvLSTM(hidden_channels, hidden_channels, kernel_size, padding))
        
        # Attention mechanism
        self.attention = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x, hidden_states=None):
        """Forward pass through all ConvLSTM layers.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_states (list): Initial hidden states for each layer
            
        Returns:
            tuple: (Output tensor, New hidden states)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
            
        output = x
        new_hidden_states = []
        
        # Process through each layer
        for i, cell in enumerate(self.cells):
            output, state = cell(output, hidden_states[i])
            new_hidden_states.append(state)
        
        # Apply attention to last time step
        last_output = output[:, -1]
        attention_weights = torch.sigmoid(self.attention(last_output))
        output_attended = last_output * attention_weights
        
        new_output = output.clone()
        new_output[:, -1] = output_attended
            
        return new_output, new_hidden_states

class ConvLSTM(nn.Module):
    """Enhanced Convolutional LSTM cell with peephole connections."""
    
    def __init__(self, in_channels, hidden_channels, kernel_size=5, padding=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Combined gates computation
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 
            hidden_channels * 4,  # 4 gates
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([hidden_channels, 42, 42])
        
        # Peephole connections
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'w_c' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, hidden_state=None):
        """Forward pass with peephole connections and layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_state (tuple): Previous (h, c) state
            
        Returns:
            tuple: (Output tensor, New state)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h_t, c_t = hidden_state
            
        # Output container
        output = torch.zeros(batch_size, seq_len, self.hidden_channels, height, width, device=x.device)
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t]
            combined = torch.cat([x_t, h_t], dim=1)
            
            # Calculate gates
            gates = self.conv(combined)
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            
            # Apply peephole connections
            i = torch.sigmoid(i + self.w_ci * c_t)
            f = torch.sigmoid(f + self.w_cf * c_t)
            g = torch.tanh(g)
            
            # Update cell state
            c_t_new = f * c_t + i * g
            c_t_new = torch.clamp(c_t_new, -10, 10)  # Prevent gradient explosion
            
            # Output gate with peephole
            o = torch.sigmoid(o + self.w_co * c_t_new)
            
            # Calculate hidden state
            h_t_new = o * torch.tanh(c_t_new)
            
            # Apply layer normalization
            if height == 42 and width == 42:
                h_t_new = self.layer_norm(h_t_new)
            
            # Store output
            output[:, t] = h_t_new
            
            # Update states
            h_t = h_t_new
            c_t = c_t_new
        
        return output, (h_t, c_t)

class HeavyPhyCRNet(nn.Module):
    """Heavy PhyCRNet with enhanced architecture for maximum performance."""
    
    def __init__(self, num_layers=12, hidden_dim=256, num_heads=8, 
                 use_attention=True, use_skip_connections=True, use_se_blocks=True,
                 dropout_rate=0.1, use_spectral_norm=False):
        super(HeavyPhyCRNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_skip_connections = use_skip_connections
        self.use_se_blocks = use_se_blocks
        
        # Multi-scale input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(4, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-scale feature extraction
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim//2, hidden_dim//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim//2, hidden_dim//2, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim//2, hidden_dim//2, kernel_size=7, padding=3),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU()
            )
        ])
        
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim + 3 * (hidden_dim//2), hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Enhanced ConvLSTM layers with attention
        self.conv_lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            # Calculate input and hidden dimensions more carefully
            if i == 0:
                input_dim = hidden_dim  # From scale fusion (256)
                layer_hidden_dim = hidden_dim  # (256)
            else:
                # Input dim is the hidden dim of the previous layer
                prev_layer_idx = i - 1
                if prev_layer_idx < num_layers//2:
                    input_dim = hidden_dim  # Previous layer outputs hidden_dim (256)
                else:
                    input_dim = hidden_dim*2  # Previous layer outputs hidden_dim*2 (512)
                
                # Current layer's hidden dimension
                if i < num_layers//2:
                    layer_hidden_dim = hidden_dim  # (256)
                else:
                    layer_hidden_dim = hidden_dim*2  # (512)
            
            # Special case: transition layer (layer 6, index 6)
            # This layer receives 256 channels but outputs 512 channels
            if i == num_layers//2:  # layer 6 (index 6)
                input_dim = hidden_dim  # 256 from previous layer
                layer_hidden_dim = hidden_dim*2  # 512 for current layer
            
            self.conv_lstm_layers.append(
                EnhancedConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=layer_hidden_dim,
                    kernel_size=3,
                    bias=True,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    use_se_block=use_se_blocks,
                    dropout_rate=dropout_rate
                )
            )
        
        # Skip connection processing
        if use_skip_connections:
            self.skip_convs = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1)  # Project to final layer size
                for i in range(num_layers//2)
            ])
        
        # Advanced output processing
        final_hidden = hidden_dim * 2  # From last layer
        self.output_processing = nn.Sequential(
            nn.Conv2d(final_hidden, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(hidden_dim//2, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(),
        )
        
        # Multi-field output heads
        self.output_heads = nn.ModuleDict({
            'velocity': nn.Sequential(
                nn.Conv2d(hidden_dim//4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=1)  # U, V
            ),
            'temperature': nn.Sequential(
                nn.Conv2d(hidden_dim//4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1)  # T
            ),
            'pressure': nn.Sequential(
                nn.Conv2d(hidden_dim//4, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=1)  # P
            )
        })
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self._apply_spectral_norm()
        
        # Initialize weights
        self._initialize_weights()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to conv layers for training stability."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)
    
    def _initialize_weights(self):
        """Initialize weights with advanced schemes."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states for all layers
        h_states = []
        c_states = []
        for i, layer in enumerate(self.conv_lstm_layers):
            # Calculate hidden dimension for this layer
            layer_hidden_dim = self.hidden_dim if i < self.num_layers//2 else self.hidden_dim*2
            h_states.append(torch.zeros(batch_size, layer_hidden_dim, height, width, 
                                      device=x.device, dtype=x.dtype))
            c_states.append(torch.zeros(batch_size, layer_hidden_dim, height, width, 
                                      device=x.device, dtype=x.dtype))
        
        # Process sequence
        skip_connections = []
        
        for t in range(seq_len):
            current_input = x[:, t]  # [batch_size, channels, height, width]
            
            # Multi-scale input processing
            processed_input = self.input_conv(current_input)
            
            # Multi-scale feature extraction
            scale_features = [processed_input]
            for scale_conv in self.scale_convs:
                scale_features.append(scale_conv(processed_input))
            
            # Fuse multi-scale features
            fused_features = torch.cat(scale_features, dim=1)
            layer_input = self.scale_fusion(fused_features)
            
            # Process through ConvLSTM layers
            for i, layer in enumerate(self.conv_lstm_layers):
                h_states[i], c_states[i] = layer(layer_input, (h_states[i], c_states[i]))
                layer_input = h_states[i]
                
                # Store skip connections from first half of layers
                if self.use_skip_connections and i < len(self.conv_lstm_layers)//2:
                    if t == seq_len - 1:  # Only store from last timestep
                        skip_connections.append(h_states[i])
        
        # Get final output
        final_output = h_states[-1]
        
        # Apply skip connections
        if self.use_skip_connections and skip_connections:
            for i, (skip, skip_conv) in enumerate(zip(skip_connections, self.skip_convs)):
                # Resize skip connection if needed
                if skip.shape != final_output.shape:
                    skip = F.interpolate(skip, size=final_output.shape[2:], 
                                       mode='bilinear', align_corners=False)
                processed_skip = skip_conv(skip)
                final_output = final_output + processed_skip
        
        # Output processing
        processed_output = self.output_processing(final_output)
        
        # Generate field-specific outputs
        velocity_output = self.output_heads['velocity'](processed_output)
        temperature_output = self.output_heads['temperature'](processed_output)
        pressure_output = self.output_heads['pressure'](processed_output)
        
        # Combine outputs
        output = torch.cat([velocity_output, temperature_output, pressure_output], dim=1)
        
        return output

class EnhancedConvLSTMCell(nn.Module):
    """Enhanced ConvLSTM cell with attention and SE blocks."""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True,
                 use_attention=True, num_heads=8, use_se_block=True, dropout_rate=0.1):
        super(EnhancedConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.use_attention = use_attention
        self.use_se_block = use_se_block
        
        # Standard ConvLSTM gates
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                                   kernel_size, padding=self.padding, bias=bias)
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(hidden_dim, num_heads)
        
        # Squeeze-and-Excitation block
        if use_se_block:
            self.se_block = SEBlock(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Layer normalization
        self.layer_norm = nn.GroupNorm(min(32, hidden_dim//4), hidden_dim)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute gates
        gates = self.conv_gates(combined)
        
        # Split gates
        i_gate, f_gate, c_gate, o_gate = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)
        
        # Update cell state
        c_next = f_gate * c_cur + i_gate * c_gate
        
        # Compute hidden state
        h_next = o_gate * torch.tanh(c_next)
        
        # Apply attention
        if self.use_attention:
            h_next = self.attention(h_next)
        
        # Apply SE block
        if self.use_se_block:
            h_next = self.se_block(h_next)
        
        # Apply normalization and dropout
        h_next = self.layer_norm(h_next)
        h_next = self.dropout(h_next)
        
        return h_next, c_next

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for ConvLSTM."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        k = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        v = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Attention computation
        attention_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(v, attention_weights.transpose(-2, -1))
        attended = attended.view(batch_size, channels, height, width)
        
        # Output projection
        output = self.output_conv(attended)
        
        return x + output  # Residual connection

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x) 