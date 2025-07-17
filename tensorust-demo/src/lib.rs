use wasm_bindgen::prelude::*;
use tensorust::{
    Tensor,
    nn::{Sequential, Linear, ReLU},
    storage::CpuStorage,
    error::Result,
};
use std::sync::Arc;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Enable console_error_panic_hook in debug mode
#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn run_basic_ops() -> String {
    set_panic_hook();
    let mut output = String::new();
    
    // Redirect stdout to our output string
    let _ = std::panic::catch_unwind(|| -> Result<()> {
        // Create tensors
        let a = Tensor::<f32, _, CpuStorage>::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
        let b = Tensor::<f32, _, CpuStorage>::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;
        
        // Perform operations
        let c = &a + &b;
        let d = &a * 2.0;
        
        // Format results
        output.push_str(&format!("a + b = {:?}\n", c.to_vec()?));
        output.push_str(&format!("a * 2 = {:?}\n", d.to_vec()?));
        
        // Matrix multiplication example
        let m1 = Tensor::<f32, _, CpuStorage>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0], 
            vec![2, 2]
        )?;
        let m2 = Tensor::<f32, _, CpuStorage>::from_vec(
            vec![5.0, 6.0, 7.0, 8.0], 
            vec![2, 2]
        )?;
        let m3 = m1.matmul(&m2)?;
        output.push_str(&format!("\nMatrix multiplication:\n{:?} @\n{:?} =\n{:?}", 
            m1.to_vec()?, m2.to_vec()?, m3.to_vec()?));
            
        Ok(())
    }).map_err(|e| {
        output = format!("Error: {:?}", e);
    });
    
    output
}

#[wasm_bindgen]
pub fn run_nn() -> String {
    set_panic_hook();
    let mut output = String::new();
    
    let _ = std::panic::catch_unwind(|| -> Result<()> {
        // Create a simple neural network
        let model = Sequential::new()
            .add(Box::new(Linear::new(2, 4)?))
            .add(Box::new(ReLU::new()))
            .add(Box::new(Linear::new(4, 1)?));
        
        // Create sample input
        let input = Tensor::<f32, _, CpuStorage>::from_vec(
            vec![0.5, -0.5], 
            vec![1, 2]
        )?;
        
        // Forward pass
        let output_tensor = model.forward(&input)?;
        
        output.push_str(&format!("Input: {:?}\n", input.to_vec()?));
        output.push_str(&format!("Neural Network Output: {:?}", output_tensor.to_vec()?));
        
        Ok(())
    }).map_err(|e| {
        output = format!("Error: {:?}", e);
    });
    
    output
}

#[wasm_bindgen]
pub fn run_autodiff() -> String {
    set_panic_hook();
    let mut output = String::new();
    
    let _ = std::panic::catch_unwind(|| -> Result<()> {
        // Create a tensor that requires gradients
        let x = Tensor::<f32, _, CpuStorage>::from_vec_with_grad(
            vec![2.0], 
            vec![1], 
            true
        )?;
        
        // Define a function: y = x^2 + 3x + 1
        let y = &x * &x + &x * 3.0 + 1.0;
        
        // Compute gradients
        y.backward()?;
        
        // Get gradient of y with respect to x
        let grad = x.grad()?;
        
        output.push_str(&format!("Function: y = x² + 3x + 1\n"));
        output.push_str(&format!("At x = 2.0, dy/dx = {}", grad[0]));
        
        Ok(())
    }).map_err(|e| {
        output = format!("Error: {:?}", e);
    });
    
    output
}
