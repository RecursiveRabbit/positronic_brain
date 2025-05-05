# Inspection Report: step3c_mlm_pair_inputs_fixed_initial.pt

Generated: 2025-05-02 15:00:36

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step3c_mlm_pair_inputs_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'mlm_pair_input_list':     [
      - [0]         {

          'original_pair':             (48, 49) (type: tuple)
          'global_target_position':             48 (type: int)
          'global_omitted_position':             49 (type: int)
          'original_token_id_at_target':             3081 (type: int)
          'mlm_input_ids': Tensor(shape=(1, 383), dtype=torch.int64, device='cpu', numel=383, data=[101, 1026, 1055, 1028, 1005, 9353, 7245, 2378, 21565, 13028, 12155, 1005, 2011, 7361, 5498, 21149, 22172, 2080, 2027, 13777...3089, 3468, 10354, 7011, 4313, 1012, 1996, 2102, 9148, 14701, 13088, 11012, 4270, 17311, 3367, 15603, 22270, 24490, 10760, 102])
          'mlm_attention_mask': Tensor(shape=(1, 383), dtype=torch.int64, device='cpu', numel=383, data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
          'mlm_mask_index':             55 (type: int)
          'tinyllama_window_ids': Tensor(shape=(304,), dtype=torch.int64, device='cpu', numel=304, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...450, 4469, 11423, 471, 263, 16403, 26195, 29889, 450, 3252, 12652, 281, 18217, 482, 471, 12243, 1233, 599, 2820, 278])
          'tinyllama_target_position_local':             48 (type: int)
          'tinyllama_omitted_position_local':             49 (type: int)
        }
      - [1]         {

          'original_pair':             (61, 62) (type: tuple)
          'global_target_position':             61 (type: int)
          'global_omitted_position':             62 (type: int)
          'original_token_id_at_target':             3099 (type: int)
          'mlm_input_ids': Tensor(shape=(1, 397), dtype=torch.int64, device='cpu', numel=397, data=[101, 1026, 1055, 1028, 1005, 9353, 7245, 2378, 21565, 13028, 12155, 1005, 2011, 7361, 5498, 21149, 22172, 2080, 2027, 13777...24490, 10760, 23467, 5685, 10521, 10230, 2618, 3993, 6777, 5178, 5897, 17311, 11365, 7028, 14399, 27241, 2869, 3762, 11253, 102])
          'mlm_attention_mask': Tensor(shape=(1, 397), dtype=torch.int64, device='cpu', numel=397, data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
          'mlm_mask_index':             73 (type: int)
          'tinyllama_window_ids': Tensor(shape=(317,), dtype=torch.int64, device='cpu', numel=317, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...482, 471, 12243, 1233, 599, 2820, 278, 2908, 322, 1320, 4350, 1319, 10757, 471, 7962, 304, 1209, 414, 491, 310])
          'tinyllama_target_position_local':             61 (type: int)
          'tinyllama_omitted_position_local':             62 (type: int)
        }
    ]
  'initial_seq_len':     862 (type: int)
  'initial_input_ids': Tensor(shape=(1, 862), dtype=torch.int64, device='cpu', numel=862, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...29889, 11511, 29892, 670, 736, 304, 19861, 2264, 2996, 2086, 5683, 304, 364, 1709, 1075, 7536, 304, 6866, 616, 29889])
  'selected_pairs':     [
      - [0]         (48, 49) (type: tuple)
      - [1]         (61, 62) (type: tuple)
    ]
  'selected_token_id':     13 (type: int)
}
```
