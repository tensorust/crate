# Tensorust Web Demo

This is a web-based demo for the Tensorust library, showcasing its capabilities through an interactive web interface powered by WebAssembly.

## Features

- **Interactive Examples**: Try out different tensor operations directly in your browser
- **Neural Network Demo**: See a simple neural network in action
- **Automatic Differentiation**: Visualize how automatic differentiation works
- **No Server Required**: Runs entirely in the browser using WebAssembly

## Prerequisites

- Rust and Cargo (latest stable version)
- `wasm-pack` (will be installed automatically if not present)
- A modern web browser with WebAssembly support (Chrome, Firefox, Safari, or Edge)

## Getting Started

1. **Build the WebAssembly module**:
   ```bash
   ./build.sh
   ```

2. **Start a local web server**:
   ```bash
   cd static
   python3 -m http.server 8000
   ```
   (You can use any HTTP server that serves static files)

3. **Open the demo**:
   Open your web browser and navigate to `http://localhost:8000`

## Examples

The demo includes the following examples:

1. **Basic Tensor Operations**:
   - Element-wise addition and multiplication
   - Matrix multiplication
   - Tensor reshaping and manipulation

2. **Neural Network**:
   - Create a simple feedforward neural network
   - Perform forward passes with sample input
   - See the network's predictions

3. **Automatic Differentiation**:
   - Define mathematical expressions
   - Compute gradients automatically
   - Visualize the computation graph

## Development

To modify or extend the demo:

1. Edit the Rust code in `src/lib.rs`
2. Update the HTML/JS in `static/index.html`
3. Rebuild with `./build.sh`
4. Refresh your browser to see changes

## License

This demo is part of the Tensorust project and is licensed under the same terms.
