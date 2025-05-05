# Inspection Report: step3d_mlm_pair_predictions_fixed_initial.pt

Generated: 2025-05-02 15:12:30

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step3d_mlm_pair_predictions_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'mlm_pair_predictions':     [
      - [0]         {

          'original_pair':             (48, 49) (type: tuple)
          'global_target_position':             48 (type: int)
          'global_omitted_position':             49 (type: int)
          'original_token_id_at_target':             3081 (type: int)
          'predicted_mlm_token_id':             3833 (type: int)
          'predicted_text':             'constructed' (type: str)
        }
      - [1]         {

          'original_pair':             (61, 62) (type: tuple)
          'global_target_position':             61 (type: int)
          'global_omitted_position':             62 (type: int)
          'original_token_id_at_target':             3099 (type: int)
          'predicted_mlm_token_id':             29337 (type: int)
          'predicted_text':             '##you' (type: str)
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
