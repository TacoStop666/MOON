import re
import matplotlib.pyplot as plt

# Ask for three log files and model names
log_files = []
model_names = []

# Baselines
log_files.append('logs\experiment_log-2025-07-29-0149-53.log') # MOON
log_files.append('logs\experiment_log-2025-08-04-0055-32.log') # FedAvg
log_files.append('logs\experiment_log-2025-08-06-0101-36.log') # FedProx

model_names.append('MOON')
model_names.append('FedAvg')
model_names.append('FedProx')

# Option for contrastive learning experiment
use_contrastive = input("Visualize contrastive learning experiment? (y/n): ").lower() == 'y'

# Model for comparing
log_files.append(input(f'Enter the path to log file: '))
model_names.append(input(f'Enter the model name for log file: '))

# Input base learning rate
base_learning_rate = input("Enter the base learning rate: ")

# Regular expression for test accuracy
test_acc_pattern = re.compile(r'Global Model Test accuracy:\s*([0-9.]+)')
after_contrastive_pattern = re.compile(r'>> After contrastive learning:')
before_contrastive_pattern = re.compile(r'>> After local training \(before contrastive learning\):')

# Store test accuracies for each model
all_test_accs = []

# input name for the figures name
figure_name = input(f'Enter the name for the figure: ')

plt.figure(figsize=(10, 6))

for idx, log_file in enumerate(log_files):
    test_accs = []
    if use_contrastive and idx == 3:
        # Special handling for contrastive learning log
        before_accs = []
        after_accs = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if before_contrastive_pattern.search(lines[i]):
                    # Look ahead for the next test accuracy
                    for j in range(i+1, min(i+5, len(lines))):
                        match = test_acc_pattern.search(lines[j])
                        if match:
                            before_accs.append(float(match.group(1)))
                            break
                if after_contrastive_pattern.search(lines[i]):
                    # Look ahead for the next test accuracy
                    for j in range(i+1, min(i+5, len(lines))):
                        match = test_acc_pattern.search(lines[j])
                        if match:
                            after_accs.append(float(match.group(1)))
                            break
                i += 1
        # Plot both before and after
        plt.plot(range(1, len(before_accs) + 1), before_accs, label=f'{model_names[idx]} Before Contrastive')
        plt.plot(range(1, len(after_accs) + 1), after_accs, label=f'{model_names[idx]} After Contrastive')
        all_test_accs.append(before_accs)
        all_test_accs.append(after_accs)
    else:
        with open(log_file, 'r') as f:
            for line in f:
                test_acc_match = test_acc_pattern.search(line)
                if test_acc_match:
                    test_accs.append(float(test_acc_match.group(1)))
        all_test_accs.append(test_accs)
        plt.plot(range(1, len(test_accs) + 1), test_accs, label=f'{model_names[idx]} Test Acc')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title(f'Global Model Test Accuracy per Epoch (Base LR: {base_learning_rate})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'result/{figure_name}.png')
# plt.show()
exit()