import json

model = json.load(open('model.json'))

params = [
    ('wte.weight', 'wte_offset'),
    ('wpe.weight', 'wpe_offset'),
    ('ln_f.weight', 'w_ln_offset'),
    ('ln_f.bias', 'b_ln_offset'),
]

for k, v in params:
    print(f"int {v} = {model[k]['data_offsets'][0] // 4}; // {k}")

block_params = [
    ('attn.c_attn.weight', 'w_attn1'),
    ('attn.c_attn.bias', 'b_attn1'),
    ('attn.c_proj.weight', 'w_attn2'),
    ('attn.c_proj.bias', 'b_attn2'),
    ('ln_1.weight', 'w_ln1'),
    ('ln_1.bias', 'b_ln1'),
    ('mlp.c_fc.weight', 'w_mlp1'),
    ('mlp.c_fc.bias', 'b_mlp1'),
    ('mlp.c_proj.weight', 'w_mlp2'),
    ('mlp.c_proj.bias', 'b_mlp2'),
    ('ln_2.weight', 'w_ln2'),
    ('ln_2.bias', 'b_ln2'),
]

for k, v in block_params:
    print(f"int {v}_offset[12] = {{ // h.*.{k}")
    for i in range(12):
        print(f"    {model[f'h.{i}.{k}']['data_offsets'][0] // 4},")
    print("};")
