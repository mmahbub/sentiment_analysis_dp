import os

dataset_name_ls     = ['imdb', 'amazon_polarity']
target_label_ls     = ['pos', 'neg']
poison_type_ls      = ['flip', 'insert', 'both']

insert_location_ls  = ['beg', 'mid_rdm', 'end']
artifact_idx_ls     = [1, 2, 3]
poison_pct_ls       = [0.5, 1, 5, 10, 20, 50]

label_dict       = {'neg': 0, 'pos': 1}

k=0
for dataset_name in dataset_name_ls[:1]:
    for target_label in target_label_ls[:1]:
        for insert_location in insert_location_ls:
            for poison_type in poison_type_ls[1:2]:
                for artifact_idx in artifact_idx_ls:
                    for poison_pct in poison_pct_ls:
                        target_label_int = label_dict[target_label]
                        change_label_to  = 1-target_label_int
                        print(dataset_name, target_label, insert_location, poison_type,
                              artifact_idx, poison_pct, target_label_int, change_label_to)
                        k+=1
                        print(k)
                        os.system(f"./data_poison.py --dataset_name {dataset_name} --poison_type {poison_type} --artifact_idx {artifact_idx} --insert_location {insert_location} --poison_pct {poison_pct} --target_label {target_label} --target_label_int {target_label_int} --change_label_to {change_label_to}; ./driver.py --dataset_name {dataset_name} --poison_type {poison_type} --artifact_idx {artifact_idx} --insert_location {insert_location} --poison_pct {poison_pct} --target_label {target_label} --target_label_int {target_label_int} --change_label_to {change_label_to} -m train --accelerator gpu --devices {4}; ./driver.py --dataset_name {dataset_name} --poison_type {poison_type} --artifact_idx {artifact_idx} --insert_location {insert_location} --poison_pct {poison_pct} --target_label {target_label} --target_label_int {target_label_int} --change_label_to {change_label_to} -m test")
