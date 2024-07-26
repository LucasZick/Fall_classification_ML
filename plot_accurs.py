import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


accuracy_df = pd.read_csv('dataset/accuracies.csv')

sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 8))

bar_plot = sns.barplot(x='Model', y='Accuracy', hue='Model', data=accuracy_df, palette='viridis')

plt.xlabel('Models', fontsize=14, weight='bold')
plt.ylabel('Accuracy', fontsize=14, weight='bold')
plt.title('Model Accuracy Comparison', fontsize=16, weight='bold')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points', 
                      fontsize=12, weight='bold')

plt.tight_layout()

plt.savefig('dataset/model_accuracy_comparison.png')

plt.show()
