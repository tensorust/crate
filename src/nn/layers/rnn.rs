//! Recurrent Neural Network (RNN) layer implementation.

use crate::{
    autodiff::{tensor::Tensor, ComputationGraph, DifferentiableOp, Node},
    tensor::Tensor as BaseTensor,
};
use std::sync::Arc;

/// A single RNN cell.
/// 
/// This implements the basic RNN computation:
/// h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
/// 
/// Where:
/// - x_t: input at time step t
/// - h_t: hidden state at time step t
/// - W_xh: input-to-hidden weights
/// - W_hh: hidden-to-hidden weights
/// - b_h: hidden bias
#[derive(Debug)]
pub struct RNNCell {
    /// Input-to-hidden weights [input_size, hidden_size]
    w_xh: Arc<Node>,
    
    /// Hidden-to-hidden weights [hidden_size, hidden_size]
    w_hh: Arc<Node>,
    
    /// Hidden bias [hidden_size]
    b_h: Option<Arc<Node>>,
    
    /// Input size
    input_size: usize,
    
    /// Hidden size
    hidden_size: usize,
    
    /// Whether to use bias
    use_bias: bool,
}

impl RNNCell {
    /// Creates a new RNN cell.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `use_bias` - Whether to use bias terms
    /// * `weight_init` - Function to initialize weights
    /// * `bias_init` - Function to initialize bias
    pub fn new(
        graph: &mut ComputationGraph,
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        weight_init: impl Fn(&[usize]) -> BaseTensor,
        bias_init: impl Fn(&[usize]) -> BaseTensor,
    ) -> Self {
        // Initialize weights
        let w_xh_data = weight_init(&[input_size, hidden_size]);
        let w_hh_data = weight_init(&[hidden_size, hidden_size]);
        
        let w_xh = graph.add_tensor(w_xh_data, true);
        let w_hh = graph.add_tensor(w_hh_data, true);
        
        // Initialize bias if needed
        let b_h = if use_bias {
            let bias_data = bias_init(&[hidden_size]);
            Some(graph.add_tensor(bias_data, true))
        } else {
            None
        };
        
        Self {
            w_xh,
            w_hh,
            b_h,
            input_size,
            hidden_size,
            use_bias,
        }
    }
    
    /// Applies the RNN cell to a single time step.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `x_t` - Input at time step t [batch_size, input_size]
    /// * `h_prev` - Previous hidden state [batch_size, hidden_size]
    /// 
    /// # Returns
    /// New hidden state [batch_size, hidden_size]
    pub fn step(
        &self,
        graph: &mut ComputationGraph,
        x_t: Arc<Node>,
        h_prev: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        // x_t @ W_xh
        let xh = graph.add_op(
            Arc::new(MatMulOp),
            &[x_t, self.w_xh.clone()],
            true,
        )?;
        
        // h_prev @ W_hh
        let hh = graph.add_op(
            Arc::new(MatMulOp),
            &[h_prev, self.w_hh.clone()],
            true,
        )?;
        
        // xh + hh
        let mut sum = graph.add_op(
            Arc::new(AddOp),
            &[xh, hh],
            true,
        )?;
        
        // Add bias if needed
        if let Some(bias) = &self.b_h {
            sum = graph.add_op(
                Arc::new(AddOp),
                &[sum, bias.clone()],
                true,
            )?;
        }
        
        // Apply tanh activation
        graph.add_op(
            Arc::new(TanhOp),
            &[sum],
            true,
        )
    }
    
    /// Returns the input-to-hidden weights.
    pub fn w_xh(&self) -> &Arc<Node> {
        &self.w_xh
    }
    
    /// Returns the hidden-to-hidden weights.
    pub fn w_hh(&self) -> &Arc<Node> {
        &self.w_hh
    }
    
    /// Returns the hidden bias if it exists.
    pub fn b_h(&self) -> Option<&Arc<Node>> {
        self.b_h.as_ref()
    }
    
    /// Returns the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    
    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    /// Returns whether the cell uses bias.
    pub fn use_bias(&self) -> bool {
        self.use_bias
    }
}

/// A multi-layer RNN that processes sequential data.
/// 
/// This implements a multi-layer RNN that processes input sequences
/// and returns the final hidden state and/or output sequences.
#[derive(Debug)]
pub struct RNN {
    /// RNN cells (one per layer)
    cells: Vec<RNNCell>,
    
    /// Number of layers
    num_layers: usize,
    
    /// Input size
    input_size: usize,
    
    /// Hidden size
    hidden_size: usize,
    
    /// Whether to return the output sequence
    return_sequences: bool,
    
    /// Whether to return the final hidden state
    return_state: bool,
}

impl RNN {
    /// Creates a new multi-layer RNN.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of RNN layers
    /// * `return_sequences` - Whether to return the output sequence
    /// * `return_state` - Whether to return the final hidden state
    /// * `use_bias` - Whether to use bias terms
    /// * `weight_init` - Function to initialize weights
    /// * `bias_init` - Function to initialize bias
    pub fn new(
        graph: &mut ComputationGraph,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        return_sequences: bool,
        return_state: bool,
        use_bias: bool,
        weight_init: impl Fn(&[usize]) -> BaseTensor,
        bias_init: impl Fn(&[usize]) -> BaseTensor,
    ) -> Self {
        let mut cells = Vec::with_capacity(num_layers);
        
        // Create the first layer
        cells.push(RNNCell::new(
            graph,
            input_size,
            hidden_size,
            use_bias,
            &weight_init,
            &bias_init,
        ));
        
        // Create subsequent layers
        for _ in 1..num_layers {
            cells.push(RNNCell::new(
                graph,
                hidden_size,  // Input size is hidden_size for deeper layers
                hidden_size,
                use_bias,
                &weight_init,
                &bias_init,
            ));
        }
        
        Self {
            cells,
            num_layers,
            input_size,
            hidden_size,
            return_sequences,
            return_state,
        }
    }
    
    /// Applies the RNN to an input sequence.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `inputs` - Input sequence of shape [batch_size, seq_len, input_size]
    /// * `initial_states` - Optional initial hidden states for each layer
    /// 
    /// # Returns
    /// A tuple containing:
    /// - The output sequence if return_sequences is true, otherwise the last output
    /// - The final hidden states if return_state is true
    pub fn forward(
        &self,
        graph: &mut ComputationGraph,
        inputs: Arc<Node>,
        initial_states: Option<Vec<Arc<Node>>>,
    ) -> Result<(Option<Arc<Node>>, Option<Vec<Arc<Node>>>), Box<dyn std::error::Error>> {
        let input_shape = inputs.tensor.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        
        // Initialize hidden states if not provided
        let mut h_prev = initial_states.unwrap_or_else(|| {
            (0..self.num_layers)
                .map(|_| {
                    graph.add_tensor(
                        BaseTensor::zeros(&[batch_size, self.hidden_size]).unwrap(),
                        false,
                    )
                })
                .collect()
        });
        
        // Transpose inputs to [seq_len, batch_size, input_size] for easier iteration
        let inputs_transposed = graph.add_op(
            Arc::new(PermuteOp::new(vec![1, 0, 2])),
            &[inputs],
            true,
        )?;
        
        let mut outputs = Vec::with_capacity(seq_len);
        
        // Process each time step
        for t in 0..seq_len {
            // Get input at time step t [batch_size, input_size]
            let x_t = graph.add_op(
                Arc::new(IndexOp::new(0, t)),
                &[inputs_transposed.clone()],
                true,
            )?;
            
            // Process through each layer
            let mut h_t = x_t;
            let mut new_h_prev = Vec::with_capacity(self.num_layers);
            
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let h_prev_layer = h_prev[layer_idx].clone();
                
                // Apply RNN cell
                let h_t_layer = cell.step(graph, h_t, h_prev_layer)?;
                
                new_h_prev.push(h_t_layer.clone());
                h_t = h_t_layer;
            }
            
            h_prev = new_h_prev;
            
            if self.return_sequences {
                outputs.push(h_t);
            }
        }
        
        // Prepare outputs
        let output_sequence = if self.return_sequences {
            // Stack outputs along time dimension [seq_len, batch_size, hidden_size]
            let stacked = graph.add_op(
                Arc::new(StackOp::new(0)),
                &outputs,
                true,
            )?;
            
            // Transpose back to [batch_size, seq_len, hidden_size]
            Some(graph.add_op(
                Arc::new(PermuteOp::new(vec![1, 0, 2])),
                &[stacked],
                true,
            )?)
        } else {
            // Just return the last output
            Some(h_prev.last().unwrap().clone())
        };
        
        let final_states = if self.return_state {
            Some(h_prev)
        } else {
            None
        };
        
        Ok((output_sequence, final_states))
    }
    
    /// Returns the RNN cells.
    pub fn cells(&self) -> &[RNNCell] {
        &self.cells
    }
    
    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    
    /// Returns the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    
    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    /// Returns whether the RNN returns sequences.
    pub fn return_sequences(&self) -> bool {
        self.return_sequences
    }
    
    /// Returns whether the RNN returns the final state.
    pub fn return_state(&self) -> bool {
        self.return_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_rnn_cell_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple RNN cell
        let input_size = 3;
        let hidden_size = 2;
        let batch_size = 2;
        
        // Initialize weights with known values for testing
        let w_xh_data = BaseTensor::from_slice(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[input_size, hidden_size],
        )?;
        
        let w_hh_data = BaseTensor::from_slice(
            &[0.1, 0.2, 0.3, 0.4],
            &[hidden_size, hidden_size],
        )?;
        
        let b_h_data = BaseTensor::from_slice(&[0.1, 0.2], &[hidden_size])?;
        
        let cell = RNNCell {
            w_xh: graph.add_tensor(w_xh_data, true),
            w_hh: graph.add_tensor(w_hh_data, true),
            b_h: Some(graph.add_tensor(b_h_data, true)),
            input_size,
            hidden_size,
            use_bias: true,
        };
        
        // Create input and initial hidden state
        let x_t_data = BaseTensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[batch_size, input_size],
        )?;
        
        let h_prev_data = BaseTensor::from_slice(
            &[0.5, 0.6, 0.7, 0.8],
            &[batch_size, hidden_size],
        )?;
        
        let x_t = graph.add_tensor(x_t_data, false);
        let h_prev = graph.add_tensor(h_prev_data, false);
        
        // Forward pass
        let h_t = cell.step(&mut graph, x_t, h_prev)?;
        
        // Check output shape
        let output_shape = h_t.tensor.shape();
        assert_eq!(output_shape, &[batch_size, hidden_size]);
        
        // Check some values
        let output_data = h_t.tensor.to_vec::<f32>()?;
        
        // Expected calculation for first sample:
        // x_t @ W_xh = [1, 2, 3] @ [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]] = [1.4, 3.2]
        // h_prev @ W_hh = [0.5, 0.6] @ [[0.1, 0.3], [0.2, 0.4]] = [0.17, 0.39]
        // sum = [1.4 + 0.17 + 0.1, 3.2 + 0.39 + 0.2] = [1.67, 3.79]
        // tanh(sum) ≈ [0.931, 0.999]
        assert_relative_eq!(output_data[0], 0.931, epsilon = 1e-3);
        assert_relative_eq!(output_data[1], 0.999, epsilon = 1e-3);
        
        Ok(())
    }
    
    #[test]
    fn test_rnn_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple RNN
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 2;
        let batch_size = 2;
        let seq_len = 3;
        
        let rnn = RNN::new(
            &mut graph,
            input_size,
            hidden_size,
            num_layers,
            true,  // return_sequences
            true,  // return_state
            true,  // use_bias
            |shape| BaseTensor::randn(shape, 0.0, 0.1),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input sequence [batch_size, seq_len, input_size]
        let input_data = BaseTensor::randn(&[batch_size, seq_len, input_size], 0.0, 1.0);
        let inputs = graph.add_tensor(input_data, true);
        
        // Forward pass
        let (output_sequence, final_states) = rnn.forward(&mut graph, inputs, None)?;
        
        // Check outputs
        if let Some(output) = output_sequence {
            let output_shape = output.tensor.shape();
            assert_eq!(output_shape, &[batch_size, seq_len, hidden_size]);
        } else {
            panic!("Expected output sequence");
        }
        
        // Check final states
        if let Some(states) = final_states {
            assert_eq!(states.len(), num_layers);
            for state in states {
                assert_eq!(state.tensor.shape(), &[batch_size, hidden_size]);
            }
        } else {
            panic!("Expected final states");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_rnn_backward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple RNN
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 1;
        let batch_size = 2;
        let seq_len = 3;
        
        let rnn = RNN::new(
            &mut graph,
            input_size,
            hidden_size,
            num_layers,
            true,  // return_sequences
            false, // return_state
            true,  // use_bias
            |shape| BaseTensor::randn(shape, 0.0, 0.1),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input sequence [batch_size, seq_len, input_size]
        let input_data = BaseTensor::randn(&[batch_size, seq_len, input_size], 0.0, 1.0);
        let inputs = graph.add_tensor(input_data, true);
        
        // Forward pass
        let (output_sequence, _) = rnn.forward(&mut graph, inputs, None)?;
        let outputs = output_sequence.unwrap();
        
        // Create a dummy loss (sum of outputs)
        let ones = graph.add_tensor(
            BaseTensor::ones(outputs.tensor.shape())?,
            false
        );
        
        let loss = graph.add_op(
            Arc::new(MatMulOp),
            &[outputs, ones],
            true,
        )?;
        
        // Backward pass
        graph.backward(&loss)?;
        
        // Check gradients
        for cell in &rnn.cells {
            assert!(cell.w_xh().gradient.is_some(), "W_xh gradient should be computed");
            assert!(cell.w_hh().gradient.is_some(), "W_hh gradient should be computed");
            if let Some(bias) = cell.b_h() {
                assert!(bias.gradient.is_some(), "Bias gradient should be computed");
            }
        }
        
        Ok(())
    }
}
