import json
from pathlib import Path
from collections import Counter

d = Path('data')

with open(d / 'taxonomy.json') as f:
    tax = json.load(f)
with open(d / 'individual_dataset.json') as f:
    records = json.load(f)
with open(d / 'splits/train.json') as f:
    train = json.load(f)
with open(d / 'splits/val.json') as f:
    val = json.load(f)
with open(d / 'splits/test.json') as f:
    test = json.load(f)
with open(d / 'sequence_dataset.json') as f:
    seqs = json.load(f)

print('=== TAXONOMY ===')
label_set = set()
for dom in tax['domains']:
    for concept in dom['concepts']:
        for m in concept['misconceptions']:
            label_set.add(m['misconception_id'])
            print(f"  Grade {dom['grade']} | {m['misconception_id']} | {m['name']}")
for b in tax['baseline_classes']:
    label_set.add(b['class_id'])
    print(f"  BASELINE | {b['class_id']}")
print(f'\nTotal taxonomy labels: {len(label_set)}')

print('\n=== INDIVIDUAL DATASET DEEP AUDIT ===')
print(f'Total records: {len(records)}')

origins = Counter(r["ground_truth"].get("data_origin", "SYNTHETIC") for r in records)
print('\nBy data origin:')
for k, v in origins.most_common():
    print(f'  {k}: {v}')

labels = Counter(r["ground_truth"]["primary_label"] for r in records)
print('\nBy label:')
for k, v in labels.most_common():
    print(f'  {k}: {v}')

has_working = sum(1 for r in records if r["student_submission"]["working_steps"])
no_working = sum(1 for r in records if not r["student_submission"]["working_steps"])
print(f'\nRecords WITH working steps: {has_working}')
print(f'Records WITHOUT working steps (UNCERTAIN tests): {no_working}')

modes = Counter(r["student_submission"]["input_mode"] for r in records)
print('\nBy input mode:')
for k, v in modes.most_common():
    print(f'  {k}: {v}')

grades = Counter(r["grade"] for r in records)
print('\nBy grade:')
for k, v in sorted(grades.items()):
    print(f'  Grade {k}: {v}')

train_grps = {r['template_group_id'] for r in train}
val_grps = {r['template_group_id'] for r in val}
test_grps = {r['template_group_id'] for r in test}
leakage = (train_grps & val_grps) | (train_grps & test_grps) | (val_grps & test_grps)
print(f'\nLeakage across splits: {leakage if leakage else "NONE"}')

print('\n=== SEQUENCE DATASET AUDIT ===')
print(f'Total sessions: {len(seqs)}')
pat = Counter(s["pattern_type"] for s in seqs)
for k, v in pat.most_common():
    print(f'  {k}: {v}')

step_counts = Counter(len(s["steps"]) for s in seqs)
print(f'Steps per session distribution: {dict(step_counts)}')

topics = Counter(s["topic_domain"] for s in seqs)
print(f'Sequence topics: {dict(topics)}')

print('\n=== SPLIT SIZES ===')
print(f'Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')

test_labels = Counter(r["ground_truth"]["primary_label"] for r in test)
print('\nTest set label distribution:')
for k, v in test_labels.most_common():
    print(f'  {k}: {v}')

sqa = [r for r in records if r['ground_truth'].get('data_origin') == 'REAL_WORLD_SCIENCEQA_BENCHMARK']
sqa_trivial = sum(1 for r in sqa if 'Selected distractor' in (r['student_submission']['working_steps'] or ''))
print(f'\nScienceQA records total: {len(sqa)}')
print(f'ScienceQA records with trivial placeholder working steps: {sqa_trivial}')

# Check if test set has label coverage from ALL major classes
all_labels = set(labels.keys())
test_label_set = set(test_labels.keys())
missing_from_test = all_labels - test_label_set
print(f'\nLabels present in full dataset: {len(all_labels)}')
print(f'Labels present in test set: {len(test_label_set)}')
print(f'Labels MISSING from test set: {missing_from_test}')
