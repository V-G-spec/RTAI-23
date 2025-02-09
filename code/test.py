# Unofficial, based on evaluate.py

import os
import csv
import time
from collections import defaultdict
from tabulate import tabulate

from analyze import analyze
from networks import get_network
from utils.loading import parse_spec
from verifier import DEVICE

networks = ['fc_base', 'fc_1', 'fc_2', 'fc_3', 'fc_4', 'fc_5', 'fc_6', 'fc_7',
            'conv_base', 'conv_1', 'conv_2', 'conv_3', 'conv_4']
test_cases_dir = 'test_cases'

# networks = ['fc_1', 'fc_2', 'fc_3', 'fc_4', 'fc_5', 'fc_6', 'fc_7',
#             'conv_1', 'conv_2', 'conv_3', 'conv_4']  # remove *_base
# test_cases_dir = 'preliminary_evaluation_test_cases'

gt = defaultdict(dict)
with open(f'{test_cases_dir}/gt.txt') as csvfile:
    for row in csv.reader(csvfile):
        net_name_, spec_, result_ = row
        gt[net_name_][spec_] = result_ == 'verified'

print(gt)

out = defaultdict(dict)
tot_time = []
for net_name in networks:
    print(f"Evaluating network {net_name}...")

    spec_dir = os.path.join(test_cases_dir, net_name)
    for i, spec in enumerate(os.listdir(spec_dir)):
        spec_path = os.path.join(spec_dir, spec).replace(
            "\\", "/")  # parse_spec can't handle '\\'

        true_label, dataset, image, eps = parse_spec(spec_path)

        net = get_network(net_name, dataset,
                          f"models/{dataset}_{net_name}.pt").to(DEVICE)
        print(net_name, '\t', f'#{i}', spec)

        image = image.to(DEVICE)
        x = net(image.unsqueeze(0))

        pred_label = x.max(dim=1)[1].item()
        assert pred_label == true_label

        start = time.time()
        try:
            # with torch.autograd.profiler.profile() as prof:
            result = analyze(net, image, eps, true_label)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
            if result:
                out[net_name][spec] = True
            else:
                out[net_name][spec] = False
        except NotImplementedError as e:
            out[net_name][spec] = repr(e)
        end = time.time()

        print("\t", f"Model took {end - start:.2f} seconds with result", out[net_name][spec],
              "" if end - start < 60 else "and timed out")
        tot_time.append(end - start)

print(out)


simplified = {'model': [], 'results': []}
for net_name in out.keys():
    table = {net_name: [], 'gt': [], 'out': [], 'result': []}
    for key in out[net_name].keys():
        table[net_name].append(key)
        table['gt'].append(gt[net_name][key])
        table['out'].append(out[net_name][key])

        if not isinstance(table['out'][-1], bool):
            result = '❔'
        elif table['gt'][-1] is True and table['out'][-1] is True:
            result = '✅'  # true positive
        elif table['gt'][-1] is False and table['out'][-1] is False:
            result = '✅'  # true negative
        elif table['gt'][-1] is True and table['out'][-1] is False:
            result = '❌'  # false negative
        elif table['gt'][-1] is False and table['out'][-1] is True:
            result = '❗'  # false positive → this is bad
        else:
            raise ValueError
        table['result'].append(result)

    simplified['model'].append(net_name)
    simplified['results'].append(', '.join(table['result']))

    print('###### ')
    print(tabulate(table, headers="keys"))
    print('######')

print(test_cases_dir)
print('###### ')
print(tabulate(simplified, headers="keys"))
print('######')
print("The evaluations individually took", tot_time, "seconds. And a combined", sum(tot_time), "seconds.")