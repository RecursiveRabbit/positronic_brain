# Positronic Brain Pipeline Analysis - long_context_sample

## Overview of Pipeline Steps

This document summarizes the key data flowing through each step of the pipeline using the long_context_sample (862 tokens) with target context size set to 500.

## Step 1: Initial Context
- **File**: step1_output_long_context_sample.pt
- **Key Data**: 
  - Initial input sequence of 862 tokens
  - Attention weights captured for next steps

## Step 2a: Attention Processing
- **File**: step2a_processed_attention_long_context_sample.pt  
- **Key Data**:
  - Processed attention scores for all tokens
  - These scores reflect the importance of each token based on model attention

## Step 2b: Brightness Calculation
- **File**: step2b_brightness_map_long_context_sample.pt
- **Key Data**:
  - Brightness values for each token position
  - Lower brightness values indicate tokens that are candidates for culling

## Step 3a: Culling Decision
- **File**: step3a_cull_decision_long_context_sample.pt
- **Key Data**:
  - cull_count: 2 (since 862 tokens > 500 target)
  - This determines how many tokens to remove

## Step 3b: Select Culling Candidates
- **File**: step3b_cull_candidates_long_context_sample.pt
- **Key Data**:
  - positions_to_cull: [63, 61]
  - These are the specific token positions selected for removal based on lowest brightness values

## Step 3c: MLM Input Preparation
- **File**: step3c_mlm_inputs_long_context_sample.pt
- **Key Data**:
  - Context windows extracted around positions 63 and 61
  - Tokenized and prepared for MLM model input with masked tokens

## Step 3d: MLM Prediction
- **File**: step3d_mlm_predictions_long_context_sample.pt
- **Key Data**:
  - Position 63: Original token ID 5131, predicted as "##you"
  - Position 61: Original token ID 3099, predicted as "##with"
  - These predictions represent what the MLM thinks should fill those positions

## Pipeline Flow Summary
1. The pipeline starts with a context of 862 tokens
2. It identifies that 2 tokens need to be culled (862 > 500)
3. It selects the lowest brightness tokens at positions 63 and 61
4. It prepares context windows around these positions
5. It uses the MLM to predict what tokens should be in those positions
6. These predictions can be used to assess the importance of the culled tokens

