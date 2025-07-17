#!/bin/bash

# Exit on error
set -e

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build the WebAssembly module
cd "$(dirname "$0")"
wasm-pack build --target web --out-name tensorust_demo --out-dir ./static

# Create a simple HTTP server to serve the demo
echo "
Demo built successfully! You can now serve the demo with:

    cd tensorust-demo/static
    python3 -m http.server 8000

Then open http://localhost:8000 in your browser.
