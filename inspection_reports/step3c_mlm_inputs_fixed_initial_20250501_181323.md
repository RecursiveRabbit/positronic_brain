# Inspection Report: step3c_mlm_inputs_fixed_initial.pt

Generated: 2025-05-01 18:13:23

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step3c_mlm_inputs_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'mlm_input_list':     [
      - [0]         {

          'global_cull_position':             63 (type: int)
          'original_token_id_at_mask':             5131 (type: int)
          'mlm_input_ids': Tensor(shape=(1, 371), dtype=torch.int64, device='cpu', numel=371, data=[101, 1026, 1055, 1028, 1005, 9353, 7245, 2378, 21565, 13028, 12155, 1005, 2011, 7361, 5498, 21149, 22172, 2080, 2027, 13777...2030, 19845, 4103, 1012, 1996, 4887, 3406, 6305, 27082, 3372, 17311, 24932, 3089, 3468, 10354, 7011, 4313, 1012, 100, 102])
          'mlm_attention_mask': Tensor(shape=(1, 371), dtype=torch.int64, device='cpu', numel=371, data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
          'mlm_mask_index':             77 (type: int)
          'tinyllama_window_ids': Tensor(shape=(319,), dtype=torch.int64, device='cpu', numel=319, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...12243, 1233, 599, 2820, 278, 2908, 322, 1320, 4350, 1319, 10757, 471, 7962, 304, 1209, 414, 491, 310, 1009, 7333])
          'tinyllama_masked_position_local':             63 (type: int)
        }
      - [1]         {

          'global_cull_position':             61 (type: int)
          'original_token_id_at_mask':             3099 (type: int)
          'mlm_input_ids': Tensor(shape=(1, 399), dtype=torch.int64, device='cpu', numel=399, data=[101, 1026, 1055, 1028, 1005, 9353, 7245, 2378, 21565, 13028, 12155, 1005, 2011, 7361, 5498, 21149, 22172, 2080, 2027, 13777...24490, 10760, 23467, 5685, 10521, 10230, 2618, 3993, 6777, 5178, 5897, 17311, 11365, 7028, 14399, 27241, 2869, 3762, 11253, 102])
          'mlm_attention_mask': Tensor(shape=(1, 399), dtype=torch.int64, device='cpu', numel=399, data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
          'mlm_mask_index':             73 (type: int)
          'tinyllama_window_ids': Tensor(shape=(317,), dtype=torch.int64, device='cpu', numel=317, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...482, 471, 12243, 1233, 599, 2820, 278, 2908, 322, 1320, 4350, 1319, 10757, 471, 7962, 304, 1209, 414, 491, 310])
          'tinyllama_masked_position_local':             61 (type: int)
        }
    ]
  'initial_seq_len':     862 (type: int)
  'initial_input_ids': Tensor(shape=(1, 862), dtype=torch.int64, device='cpu', numel=862, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...29889, 11511, 29892, 670, 736, 304, 19861, 2264, 2996, 2086, 5683, 304, 364, 1709, 1075, 7536, 304, 6866, 616, 29889])
  'positions_to_cull':     [
      - [0]         63 (type: int)
      - [1]         61 (type: int)
    ]
  'selected_token_id':     13 (type: int)
}
```
