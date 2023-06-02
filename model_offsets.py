import json

model = json.load(open('model.json'))

params = [
    ('wte.weight', 'wt'),
    ('wpe.weight', 'wp'),
    ('ln_f.weight', 'g0'),
    ('ln_f.bias', 'c0'),
]

for k, v in params:
    print(f"int {v} = {model[k]['data_offsets'][0] // 4}; // {k}")

block_params = [
    ('attn.c_attn.weight', 'w1'),
    ('attn.c_attn.bias', 'b1'),
    ('attn.c_proj.weight', 'w2'),
    ('attn.c_proj.bias', 'b2'),
    ('ln_1.weight', 'g1'),
    ('ln_1.bias', 'c1'),
    ('mlp.c_fc.weight', 'w3'),
    ('mlp.c_fc.bias', 'b3'),
    ('mlp.c_proj.weight', 'w4'),
    ('mlp.c_proj.bias', 'b4'),
    ('ln_2.weight', 'g2'),
    ('ln_2.bias', 'c2'),
]

for k, v in block_params:
    print(f"int {v}[12] = {{ // h.*.{k}")
    for i in range(12):
        print(f"    {model[f'h.{i}.{k}']['data_offsets'][0] // 4},")
    print("};")
