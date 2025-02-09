# Translate evaluate to Python so it is OS-agnostic

import os
import subprocess

networks = ['fc_base', 'fc_1', 'fc_2', 'fc_3', 'fc_4', 'fc_5', 'fc_6', 'fc_7',
            'conv_base', 'conv_1', 'conv_2', 'conv_3', 'conv_4']
# networks = ['fc_base']
test_cases_dir = 'test_cases'

networks = ['conv_4']  # remove *_base
test_cases_dir = 'preliminary_evaluation_test_cases'

for net in networks:
    print(f"Evaluating network {net}...")

    spec_dir = os.path.join(test_cases_dir, net)
    for spec in os.listdir(spec_dir):
        spec_path = os.path.join(spec_dir, spec).replace("\\", "/")  # parse_spec can't handle '\\'

        # Execute the python command
        print(" ".join(["python", "code/verifier.py", "--net", net, "--spec", spec_path]))
        subprocess.run(["python", "code/verifier.py", "--net", net, "--spec", spec_path])
