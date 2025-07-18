//! Neural network layers for Tensorust.
//!
//! This module provides various neural network layers that can be composed
//! to build complex models. Each layer implements the forward and backward
//! passes needed for training with automatic differentiation.

pub mod batch_norm;
pub mod conv2d;
pub mod dense;
pub mod dropout;
pub mod lstm;
pub mod rnn;
