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

# Model for comparing
log_files.append(input(f'Enter the path to log file: '))
model_names.append(input(f'Enter the model name for log file: '))

# Regular expression for test accuracy
test_acc_pattern = re.compile(r'Global Model Test accuracy:\s*([0-9.]+)')

# Store test accuracies for each model
all_test_accs = []

# input name for the figures name
figure_name = input(f'Enter the name for the figure: ')

for log_file in log_files:
    test_accs = []
    with open(log_file, 'r') as f:
        for line in f:
            test_acc_match = test_acc_pattern.search(line)
            if test_acc_match:
                test_accs.append(float(test_acc_match.group(1)))
    all_test_accs.append(test_accs)

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(range(1, len(all_test_accs[i]) + 1), all_test_accs[i], label=f'{model_names[i]} Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Global Model Test Accuracy per Round')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'result/{figure_name}.png')
plt.show()