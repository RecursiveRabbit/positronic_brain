# Inspection Report: step1_output_fixed_initial.pt

Generated: 2025-05-01 18:00:40

File path: `/home/evans/Coding_Projects/positronic_brain/tests/captures/step1_output_fixed_initial.pt`

## File Contents

```
{

  'test_id':     'fixed_initial' (type: str)
  'initial_input_ids': Tensor(shape=(1, 13), dtype=torch.int64, device='cuda:0', numel=13, data=[[1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889]])
  'initial_attention_mask': Tensor(shape=(1, 13), dtype=torch.int64, device='cuda:0', numel=13, data=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
  'initial_seq_len':     13 (type: int)
  'step1_input_token': Tensor(shape=(1, 1), dtype=torch.int64, device='cuda:0', numel=1, data=[[29889]])
  'step1_position_ids': Tensor(shape=(1, 1), dtype=torch.int64, device='cuda:0', numel=1, data=[[12]])
  'logits': Tensor(shape=(1, 1, 32000), dtype=torch.float32, device='cuda:0', numel=32000, data=<large tensor>)
  'attentions':     (tensor([[[[0.0058, 0.0655, 0.0553, 0.0461, 0.0391, 0.0251, 0.0357, 0.0345,...1, 2.4385e-01, 1.9635e-01]]]], device='cuda:0',
       requires_grad=True)) (type: tuple)
  'selected_token_id':     13 (type: int)
  'next_kv_cache':     <transformers.cache_utils.DynamicCache object at 0x79e65cf8f9d0> (type: DynamicCache)
  'token_probs': Tensor(shape=(1, 32000), dtype=torch.float32, device='cuda:0', numel=32000, data=<large tensor>)
  'top_tokens_info':     [
      - [0]         {

          'token_id':             13 (type: int)
          'probability':             0.7216 (type: float)
        }
      - [1]         {

          'token_id':             29871 (type: int)
          'probability':             0.0508 (type: float)
        }
      - [2]         {

          'token_id':             450 (type: int)
          'probability':             0.0372 (type: float)
        }
      - [3]         {

          'token_id':             2 (type: int)
          'probability':             0.0218 (type: float)
        }
      - [4]         {

          'token_id':             910 (type: int)
          'probability':             0.0142 (type: float)
        }
      - [5]         {

          'token_id':             306 (type: int)
          'probability':             0.0086 (type: float)
        }
      - [6]         {

          'token_id':             319 (type: int)
          'probability':             0.0078 (type: float)
        }
      - [7]         {

          'token_id':             313 (type: int)
          'probability':             0.0078 (type: float)
        }
      - [8]         {

          'token_id':             739 (type: int)
          'probability':             0.0067 (type: float)
        }
      - [9]         {

          'token_id':             891 (type: int)
          'probability':             0.0039 (type: float)
        }
    ]
}
```
