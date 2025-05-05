# Inspection Report: step3e_actions_fixed_initial.pt

Generated: 2025-05-01 18:13:26

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step3e_actions_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'action_list':     [
      - [0]         {

          'action':             'delete' (type: str)
          'position':             63 (type: int)
        }
      - [1]         {

          'action':             'delete' (type: str)
          'position':             61 (type: int)
        }
      - [2]         {

          'action':             'replace' (type: str)
          'position':             63 (type: int)
          'new_token_ids':             [
              - [0]                 366 (type: int)
            ]
        }
      - [3]         {

          'action':             'replace' (type: str)
          'position':             61 (type: int)
          'new_token_ids':             [
              - [0]                 411 (type: int)
            ]
        }
      - [4]         {

          'action':             'add' (type: str)
          'token_id':             13 (type: int)
        }
    ]
  'initial_input_ids': Tensor(shape=(1, 862), dtype=torch.int64, device='cpu', numel=862, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...29889, 11511, 29892, 670, 736, 304, 19861, 2264, 2996, 2086, 5683, 304, 364, 1709, 1075, 7536, 304, 6866, 616, 29889])
  'initial_seq_len':     862 (type: int)
  'positions_to_cull':     [
      - [0]         63 (type: int)
      - [1]         61 (type: int)
    ]
  'mlm_predictions':     [
      - [0]         {

          'global_cull_position':             63 (type: int)
          'original_token_id_at_mask':             5131 (type: int)
          'predicted_mlm_token_id':             29337 (type: int)
          'predicted_text':             '##you' (type: str)
        }
      - [1]         {

          'global_cull_position':             61 (type: int)
          'original_token_id_at_mask':             3099 (type: int)
          'predicted_mlm_token_id':             24415 (type: int)
          'predicted_text':             '##with' (type: str)
        }
    ]
}
```
