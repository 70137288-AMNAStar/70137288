import pandas as pd
import math
from collections import Counter

# DATASET: Employee Promotion

data = {
    'Experience': ['Low','Medium','High','High','Medium','Low','Medium',
                   'High','Low','Medium','High','Low','Medium','High'],
    'Performance': ['Poor','Average','Good','Excellent','Good','Poor','Average',
                    'Excellent','Poor','Good','Good','Poor','Average','Excellent'],
    'Education': ['Diploma','Bachelor','Bachelor','Master','Bachelor','Diploma','Bachelor',
                  'Master','Diploma','Bachelor','Bachelor','Diploma','Bachelor','Master'],
    'Certifications': ['No','Yes','Yes','Yes','No','No','Yes',
                        'Yes','No','Yes','No','No','Yes','Yes'],
    'Promotion': ['No','No','Yes','Yes','Yes','No','Yes',
                  'Yes','No','Yes','Yes','No','Yes','Yes']
}

df = pd.DataFrame(data)


def entropy(data):
    labels = data.iloc[:, -1]
    counts = Counter(labels)
    total = len(labels)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def information_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy


def id3(data, attributes):
    labels = data.iloc[:, -1]

    if len(set(labels)) == 1:
        return labels.iloc[0]

    if len(attributes) == 0:
        return labels.mode()[0]

    gains = {attr: information_gain(data, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = id3(subset, remaining_attrs)

    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    value = sample[attr]

    return predict(tree[attr][value], sample)


attributes = list(df.columns[:-1])
decision_tree = id3(df, attributes)

print("Decision Tree:")
print(decision_tree)


sample = {
    'Experience': 'High',
    'Performance': 'Good',
    'Education': 'Bachelor',
    'Certifications': 'Yes'
}

print("\nPrediction for sample:")
print(predict(decision_tree, sample))
