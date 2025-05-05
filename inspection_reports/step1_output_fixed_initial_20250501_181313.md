# Inspection Report: step1_output_fixed_initial.pt

Generated: 2025-05-01 18:13:14

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step1_output_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'initial_input_ids': Tensor(shape=(1, 862), dtype=torch.int64, device='cuda:0', numel=862, data=[1, 525, 29909, 19785, 262, 1754, 363, 1023, 29915, 491, 6461, 262, 291, 630, 29902, 6720, 13, 15597, 892, 263...29889, 11511, 29892, 670, 736, 304, 19861, 2264, 2996, 2086, 5683, 304, 364, 1709, 1075, 7536, 304, 6866, 616, 29889])
  'initial_attention_mask': Tensor(shape=(1, 862), dtype=torch.int64, device='cuda:0', numel=862, data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  'initial_seq_len':     862 (type: int)
  'step1_input_token': Tensor(shape=(1, 1), dtype=torch.int64, device='cuda:0', numel=1, data=[[29889]])
  'step1_position_ids': Tensor(shape=(1, 1), dtype=torch.int64, device='cuda:0', numel=1, data=[[861]])
  'logits': Tensor(shape=(1, 1, 32000), dtype=torch.float32, device='cuda:0', numel=32000, data=<large tensor>)
  'attentions':     (tensor([[[[0.0008, 0.0001, 0.0002,  ..., 0.0065, 0.0366, 0.0366]],

      ...          7.7347e-02, 6.5355e-02]]]], device='cuda:0', requires_grad=True)) (type: tuple)
  'selected_token_id':     13 (type: int)
  'next_kv_cache':     <transformers.cache_utils.DynamicCache object at 0x717ffbbb75d0> (type: DynamicCache)
  'token_probs': Tensor(shape=(1, 32000), dtype=torch.float32, device='cuda:0', numel=32000, data=<large tensor>)
  'top_tokens_info':     [
      - [0]         {

          'token_id':             13 (type: int)
          'probability':             0.436 (type: float)
        }
      - [1]         {

          'token_id':             940 (type: int)
          'probability':             0.1043 (type: float)
        }
      - [2]         {

          'token_id':             450 (type: int)
          'probability':             0.0642 (type: float)
        }
      - [3]         {

          'token_id':             1932 (type: int)
          'probability':             0.0237 (type: float)
        }
      - [4]         {

          'token_id':             512 (type: int)
          'probability':             0.0219 (type: float)
        }
      - [5]         {

          'token_id':             319 (type: int)
          'probability':             0.0202 (type: float)
        }
      - [6]         {

          'token_id':             3600 (type: int)
          'probability':             0.018 (type: float)
        }
      - [7]         {

          'token_id':             739 (type: int)
          'probability':             0.0146 (type: float)
        }
      - [8]         {

          'token_id':             2973 (type: int)
          'probability':             0.0126 (type: float)
        }
      - [9]         {

          'token_id':             2860 (type: int)
          'probability':             0.0114 (type: float)
        }
    ]
}
```
