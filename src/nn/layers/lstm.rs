//! Long Short-Term Memory (LSTM) layer implementation.
//!
//! This module implements the LSTM layer, which is a type of recurrent neural network
//! that can learn long-term dependencies in sequential data.

use crate::{
    autodiff::{tensor::Tensor, ComputationGraph, DifferentiableOp, Node},
    tensor::Tensor as BaseTensor,
};
use std::sync::Arc;

/// A single LSTM cell.
///
/// This implements the standard LSTM equations:
/// - Input gate: i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
/// - Forget gate: f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
/// - Cell gate: g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
/// - Output gate: o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
/// - Cell state: c_t = f_t * c_{t-1} + i_t * g_t
/// - Hidden state: h_t = o_t * tanh(c_t)
#[derive(Debug)]
pub struct LSTMCell {
    // Input weights and biases
    w_ii: Arc<Node>, // Input gate weights [input_size, hidden_size]
    w_if: Arc<Node>, // Forget gate weights [input_size, hidden_size]
    w_ig: Arc<Node>, // Cell gate weights [input_size, hidden_size]
    w_io: Arc<Node>, // Output gate weights [input_size, hidden_size]
    
    // Hidden weights and biases
    w_hi: Arc<Node>, // Input gate hidden weights [hidden_size, hidden_size]
    w_hf: Arc<Node>, // Forget gate hidden weights [hidden_size, hidden_size]
    w_hg: Arc<Node>, // Cell gate hidden weights [hidden_size, hidden_size]
    w_ho: Arc<Node>, // Output gate hidden weights [hidden_size, hidden_size]
    
    // Biases
    b_ii: Option<Arc<Node>>, // Input gate bias [hidden_size]
    b_if: Option<Arc<Node>>, // Forget gate bias [hidden_size]
    b_ig: Option<Arc<Node>>, // Cell gate bias [hidden_size]
    b_io: Option<Arc<Node>>, // Output gate bias [hidden_size]
    b_hi: Option<Arc<Node>>, // Input gate hidden bias [hidden_size]
    b_hf: Option<Arc<Node>>, // Forget gate hidden bias [hidden_size]
    b_hg: Option<Arc<Node>>, // Cell gate hidden bias [hidden_size]
    b_ho: Option<Arc<Node>>, // Output gate hidden bias [hidden_size]
    
    // Layer parameters
    input_size: usize,
    hidden_size: usize,
    use_bias: bool,
}

impl LSTMCell {
    /// Creates a new LSTM cell.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `use_bias` - Whether to use bias terms
    /// * `weight_init` - Function to initialize weights
    /// * `bias_init` - Function to initialize biases
    pub fn new<F, G>(
        graph: &mut ComputationGraph,
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        weight_init: F,
        bias_init: G,
    ) -> Self
    where
        F: Fn(&[usize]) -> BaseTensor,
        G: Fn(&[usize]) -> BaseTensor,
    {
        // Initialize input weights
        let w_ii = graph.add_tensor(
            weight_init(&[input_size, hidden_size]),
            true,
        );
        
        let w_if = graph.add_tensor(
            weight_init(&[input_size, hidden_size]),
            true,
        );
        
        let w_ig = graph.add_tensor(
            weight_init(&[input_size, hidden_size]),
            true,
        );
        
        let w_io = graph.add_tensor(
            weight_init(&[input_size, hidden_size]),
            true,
        );
        
        // Initialize hidden weights
        let w_hi = graph.add_tensor(
            weight_init(&[hidden_size, hidden_size]),
            true,
        );
        
        let w_hf = graph.add_tensor(
            weight_init(&[hidden_size, hidden_size]),
            true,
        );
        
        let w_hg = graph.add_tensor(
            weight_init(&[hidden_size, hidden_size]),
            true,
        );
        
        let w_ho = graph.add_tensor(
            weight_init(&[hidden_size, hidden_size]),
            true,
        );
        
        // Initialize biases if needed
        let (b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho) = if use_bias {
            (
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
                Some(graph.add_tensor(bias_init(&[hidden_size]), true)),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };
        
        Self {
            w_ii, w_if, w_ig, w_io,
            w_hi, w_hf, w_hg, w_ho,
            b_ii, b_if, b_ig, b_io,
            b_hi, b_hf, b_hg, b_ho,
            input_size,
            hidden_size,
            use_bias,
        }
    }
    
    /// Applies the LSTM cell to a single time step.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `x_t` - Input at time step t [batch_size, input_size]
    /// * `h_prev` - Previous hidden state [batch_size, hidden_size]
    /// * `c_prev` - Previous cell state [batch_size, hidden_size]
    /// 
    /// # Returns
    /// A tuple containing:
    /// - New hidden state [batch_size, hidden_size]
    /// - New cell state [batch_size, hidden_size]
    pub fn step(
        &self,
        graph: &mut ComputationGraph,
        x_t: Arc<Node>,
        h_prev: Arc<Node>,
        c_prev: Arc<Node>,
    ) -> Result<(Arc<Node>, Arc<Node>), Box<dyn std::error::Error>> {
        // Input gate: i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
        let i_t = self.gate_forward(
            graph, &x_t, &h_prev,
            &self.w_ii, &self.w_hi,
            self.b_ii.as_ref(), self.b_hi.as_ref(),
            true, // sigmoid
        )?;
        
        // Forget gate: f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
        let f_t = self.gate_forward(
            graph, &x_t, &h_prev,
            &self.w_if, &self.w_hf,
            self.b_if.as_ref(), self.b_hf.as_ref(),
            true, // sigmoid
        )?;
        
        // Cell gate: g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
        let g_t = self.gate_forward(
            graph, &x_t, &h_prev,
            &self.w_ig, &self.w_hg,
            self.b_ig.as_ref(), self.b_hg.as_ref(),
            false, // tanh
        )?;
        
        // Output gate: o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
        let o_t = self.gate_forward(
            graph, &x_t, &h_prev,
            &self.w_io, &self.w_ho,
            self.b_io.as_ref(), self.b_ho.as_ref(),
            true, // sigmoid
        )?;
        
        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let f_c = graph.add_op(
            Arc::new(MultiplyOp),
            &[f_t, c_prev],
            true,
        )?;
        
        let i_g = graph.add_op(
            Arc::new(MultiplyOp),
            &[i_t, g_t],
            true,
        )?;
        
        let c_t = graph.add_op(
            Arc::new(AddOp),
            &[f_c, i_g],
            true,
        )?;
        
        // Hidden state: h_t = o_t * tanh(c_t)
        let c_t_tanh = graph.add_op(
            Arc::new(TanhOp),
            &[c_t.clone()],
            true,
        )?;
        
        let h_t = graph.add_op(
            Arc::new(MultiplyOp),
            &[o_t, c_t_tanh],
            true,
        )?;
        
        Ok((h_t, c_t))
    }
    
    /// Helper function to compute a single gate's output.
    fn gate_forward(
        &self,
        graph: &mut ComputationGraph,
        x_t: &Arc<Node>,
        h_prev: &Arc<Node>,
        w_x: &Arc<Node>,
        w_h: &Arc<Node>,
        b_x: Option<&Arc<Node>>,
        b_h: Option<&Arc<Node>>,
        use_sigmoid: bool,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        // Linear transformation for input
        let x_out = graph.add_op(
            Arc::new(MatMulOp),
            &[x_t.clone(), w_x.clone()],
            true,
        )?;
        
        // Linear transformation for hidden state
        let h_out = graph.add_op(
            Arc::new(MatMulOp),
            &[h_prev.clone(), w_h.clone()],
            true,
        )?;
        
        // Sum the transformations
        let mut sum = graph.add_op(
            Arc::new(AddOp),
            &[x_out, h_out],
            true,
        )?;
        
        // Add biases if they exist
        if let (Some(bx), Some(bh)) = (b_x, b_h) {
            // Broadcast bias to match batch size
            let bx_broadcast = graph.add_op(
                Arc::new(ExpandOp::new(vec![0, self.hidden_size])),
                &[bx.clone()],
                true,
            )?;
            
            let bh_broadcast = graph.add_op(
                Arc::new(ExpandOp::new(vec![0, self.hidden_size])),
                &[bh.clone()],
                true,
            )?;
            
            sum = graph.add_op(
                Arc::new(AddOp),
                &[sum, bx_broadcast],
                true,
            )?;
            
            sum = graph.add_op(
                Arc::new(AddOp),
                &[sum, bh_broadcast],
                true,
            )?;
        }
        
        // Apply activation function
        if use_sigmoid {
            graph.add_op(
                Arc::new(SigmoidOp),
                &[sum],
                true,
            )
        } else {
            graph.add_op(
                Arc::new(TanhOp),
                &[sum],
                true,
            )
        }
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

/// A multi-layer LSTM that processes sequential data.
///
/// This implements a multi-layer LSTM that processes input sequences
/// and returns the final hidden state and/or output sequences.
#[derive(Debug)]
pub struct LSTM {
    /// LSTM cells (one per layer)
    cells: Vec<LSTMCell>,
    
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
    
    /// Whether to return the final cell state
    return_cell_state: bool,
}

impl LSTM {
    /// Creates a new multi-layer LSTM.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input_size` - Size of input features
    /// * `hidden_size` - Size of hidden state
    /// * `num_layers` - Number of LSTM layers
    /// * `return_sequences` - Whether to return the output sequence
    /// * `return_state` - Whether to return the final hidden state
    /// * `return_cell_state` - Whether to return the final cell state
    /// * `use_bias` - Whether to use bias terms
    /// * `weight_init` - Function to initialize weights
    /// * `bias_init` - Function to initialize biases
    pub fn new<F, G>(
        graph: &mut ComputationGraph,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        return_sequences: bool,
        return_state: bool,
        return_cell_state: bool,
        use_bias: bool,
        weight_init: F,
        bias_init: G,
    ) -> Self
    where
        F: Fn(&[usize]) -> BaseTensor + Copy,
        G: Fn(&[usize]) -> BaseTensor + Copy,
    {
        let mut cells = Vec::with_capacity(num_layers);
        
        // Create the first layer
        cells.push(LSTMCell::new(
            graph,
            input_size,
            hidden_size,
            use_bias,
            weight_init,
            bias_init,
        ));
        
        // Create subsequent layers
        for _ in 1..num_layers {
            cells.push(LSTMCell::new(
                graph,
                hidden_size,  // Input size is hidden_size for deeper layers
                hidden_size,
                use_bias,
                weight_init,
                bias_init,
            ));
        }
        
        Self {
            cells,
            num_layers,
            input_size,
            hidden_size,
            return_sequences,
            return_state,
            return_cell_state,
        }
    }
    
    /// Applies the LSTM to an input sequence.
    /// 
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `inputs` - Input sequence of shape [batch_size, seq_len, input_size]
    /// * `initial_states` - Optional initial hidden and cell states for each layer
    ///
    /// # Returns
    /// A tuple containing:
    /// - The output sequence if return_sequences is true, otherwise the last output
    /// - The final hidden states if return_state is true
    /// - The final cell states if return_cell_state is true
    pub fn forward(
        &self,
        graph: &mut ComputationGraph,
        inputs: Arc<Node>,
        initial_states: Option<(Vec<Arc<Node>>, Vec<Arc<Node>>)>,
    ) -> Result<(Option<Arc<Node>>, Option<Vec<Arc<Node>>>, Option<Vec<Arc<Node>>>>, Box<dyn std::error::Error>> {
        let input_shape = inputs.tensor.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        
        // Initialize hidden and cell states if not provided
        let (mut h_prev, mut c_prev) = if let Some((h, c)) = initial_states {
            (h, c)
        } else {
            let h: Vec<_> = (0..self.num_layers)
                .map(|_| {
                    graph.add_tensor(
                        BaseTensor::zeros(&[batch_size, self.hidden_size]).unwrap(),
                        false,
                    )
                })
                .collect();
                
            let c: Vec<_> = (0..self.num_layers)
                .map(|_| {
                    graph.add_tensor(
                        BaseTensor::zeros(&[batch_size, self.hidden_size]).unwrap(),
                        false,
                    )
                })
                .collect();
                
            (h, c)
        };
        
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
            let mut new_c_prev = Vec::with_capacity(self.num_layers);
            
            for (layer_idx, cell) in self.cells.iter().enumerate() {
                let h_prev_layer = h_prev[layer_idx].clone();
                let c_prev_layer = c_prev[layer_idx].clone();
                
                // Apply LSTM cell
                let (h_t_layer, c_t_layer) = cell.step(
                    graph,
                    h_t,
                    h_prev_layer,
                    c_prev_layer,
                )?;
                
                new_h_prev.push(h_t_layer.clone());
                new_c_prev.push(c_t_layer.clone());
                h_t = h_t_layer;
            }
            
            h_prev = new_h_prev;
            c_prev = new_c_prev;
            
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
        
        let final_hidden = if self.return_state {
            Some(h_prev)
        } else {
            None
        };
        
        let final_cell = if self.return_cell_state {
            Some(c_prev)
        } else {
            None
        };
        
        Ok((output_sequence, final_hidden, final_cell))
    }
    
    /// Returns the LSTM cells.
    pub fn cells(&self) -> &[LSTMCell] {
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
    
    /// Returns whether the LSTM returns sequences.
    pub fn return_sequences(&self) -> bool {
        self.return_sequences
    }
    
    /// Returns whether the LSTM returns the final hidden state.
    pub fn return_state(&self) -> bool {
        self.return_state
    }
    
    /// Returns whether the LSTM returns the final cell state.
    pub fn return_cell_state(&self) -> bool {
        self.return_cell_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_lstm_cell_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple LSTM cell with known weights for testing
        let input_size = 3;
        let hidden_size = 2;
        let batch_size = 2;
        
        // Initialize weights with known values for testing
        let init_weights = |shape: &[usize]| {
            // Simple identity-like initialization for testing
            let size = shape.iter().product();
            let mut data = vec![0.1; size];
            
            // Make the weights somewhat identity-like for stable testing
            if shape.len() == 2 && shape[0] == shape[1] {
                // Hidden weights - make them identity-like
                for i in 0..shape[0].min(shape[1]) {
                    data[i * shape[1] + i] = 1.0;
                }
            }
            
            BaseTensor::from_slice(&data, shape).unwrap()
        };
        
        let init_biases = |shape: &[usize]| {
            // Small biases for testing
            BaseTensor::zeros(shape).unwrap()
        };
        
        let cell = LSTMCell::new(
            &mut graph,
            input_size,
            hidden_size,
            true, // use_bias
            init_weights,
            init_biases,
        );
        
        // Create input and initial states
        let x_t_data = BaseTensor::from_slice(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[batch_size, input_size],
        )?;
        
        let h_prev_data = BaseTensor::from_slice(
            &[0.1, 0.2, 0.3, 0.4],
            &[batch_size, hidden_size],
        )?;
        
        let c_prev_data = BaseTensor::from_slice(
            &[0.05, 0.1, 0.15, 0.2],
            &[batch_size, hidden_size],
        )?;
        
        let x_t = graph.add_tensor(x_t_data, false);
        let h_prev = graph.add_tensor(h_prev_data, false);
        let c_prev = graph.add_tensor(c_prev_data, false);
        
        // Forward pass
        let (h_t, c_t) = cell.step(&mut graph, x_t, h_prev, c_prev)?;
        
        // Check output shapes
        assert_eq!(h_t.tensor.shape(), &[batch_size, hidden_size]);
        assert_eq!(c_t.tensor.shape(), &[batch_size, hidden_size]);
        
        // Check that the outputs are different from inputs (sanity check)
        let h_t_data = h_t.tensor.to_vec::<f32>()?;
        let c_t_data = c_t.tensor.to_vec::<f32>()?;
        
        assert_ne!(h_t_data, h_prev_data.to_vec::<f32>()?);
        assert_ne!(c_t_data, c_prev_data.to_vec::<f32>()?);
        
        // Check that the outputs have reasonable values (not NaN or infinity)
        for &val in h_t_data.iter().chain(c_t_data.iter()) {
            assert!(!val.is_nan());
            assert!(val.is_finite());
        }
        
        Ok(())
    }
    
    #[test]
    fn test_lstm_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple LSTM
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 2;
        let batch_size = 2;
        let seq_len = 3;
        
        let lstm = LSTM::new(
            &mut graph,
            input_size,
            hidden_size,
            num_layers,
            true,  // return_sequences
            true,  // return_state
            true,  // return_cell_state
            true,  // use_bias
            |shape| BaseTensor::randn(shape, 0.0, 0.1),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input sequence [batch_size, seq_len, input_size]
        let input_data = BaseTensor::randn(&[batch_size, seq_len, input_size], 0.0, 1.0);
        let inputs = graph.add_tensor(input_data, true);
        
        // Forward pass
        let (output_sequence, final_hidden, final_cell) = lstm.forward(&mut graph, inputs, None)?;
        
        // Check outputs
        if let Some(output) = output_sequence {
            let output_shape = output.tensor.shape();
            assert_eq!(output_shape, &[batch_size, seq_len, hidden_size]);
        } else {
            panic!("Expected output sequence");
        }
        
        // Check final hidden states
        if let Some(states) = final_hidden {
            assert_eq!(states.len(), num_layers);
            for state in states {
                assert_eq!(state.tensor.shape(), &[batch_size, hidden_size]);
            }
        } else {
            panic!("Expected final hidden states");
        }
        
        // Check final cell states
        if let Some(states) = final_cell {
            assert_eq!(states.len(), num_layers);
            for state in states {
                assert_eq!(state.tensor.shape(), &[batch_size, hidden_size]);
            }
        } else {
            panic!("Expected final cell states");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_lstm_backward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple LSTM
        let input_size = 3;
        let hidden_size = 2;
        let num_layers = 1;
        let batch_size = 2;
        let seq_len = 3;
        
        let lstm = LSTM::new(
            &mut graph,
            input_size,
            hidden_size,
            num_layers,
            true,  // return_sequences
            false, // return_state
            false, // return_cell_state
            true,  // use_bias
            |shape| BaseTensor::randn(shape, 0.0, 0.1),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input sequence [batch_size, seq_len, input_size]
        let input_data = BaseTensor::randn(&[batch_size, seq_len, input_size], 0.0, 1.0);
        let inputs = graph.add_tensor(input_data, true);
        
        // Forward pass
        let (output_sequence, _, _) = lstm.forward(&mut graph, inputs.clone(), None)?;
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
        for cell in &lstm.cells {
            // Check input weights
            assert!(cell.w_ii.gradient.is_some(), "W_ii gradient should be computed");
            assert!(cell.w_if.gradient.is_some(), "W_if gradient should be computed");
            assert!(cell.w_ig.gradient.is_some(), "W_ig gradient should be computed");
            assert!(cell.w_io.gradient.is_some(), "W_io gradient should be computed");
            
            // Check hidden weights
            assert!(cell.w_hi.gradient.is_some(), "W_hi gradient should be computed");
            assert!(cell.w_hf.gradient.is_some(), "W_hf gradient should be computed");
            assert!(cell.w_hg.gradient.is_some(), "W_hg gradient should be computed");
            assert!(cell.w_ho.gradient.is_some(), "W_ho gradient should be computed");
            
            // Check biases if they exist
            if cell.use_bias {
                assert!(cell.b_ii.as_ref().unwrap().gradient.is_some(), "b_ii gradient should be computed");
                assert!(cell.b_if.as_ref().unwrap().gradient.is_some(), "b_if gradient should be computed");
                assert!(cell.b_ig.as_ref().unwrap().gradient.is_some(), "b_ig gradient should be computed");
                assert!(cell.b_io.as_ref().unwrap().gradient.is_some(), "b_io gradient should be computed");
                assert!(cell.b_hi.as_ref().unwrap().gradient.is_some(), "b_hi gradient should be computed");
                assert!(cell.b_hf.as_ref().unwrap().gradient.is_some(), "b_hf gradient should be computed");
                assert!(cell.b_hg.as_ref().unwrap().gradient.is_some(), "b_hg gradient should be computed");
                assert!(cell.b_ho.as_ref().unwrap().gradient.is_some(), "b_ho gradient should be computed");
            }
        }
        
        // Check input gradient
        assert!(inputs.gradient.is_some(), "Input gradient should be computed");
        
        Ok(())
    }
}
