import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Kontingenztabelle erstellen
table = np.zeros((2, 2))

# Vergleiche die Vorhersagen für A und als Alternative B
for A_pred, B_pred, true_label in zip(y_pred_A, y_pred_B, y_test):
    if A_pred == true_label and B_pred == true_label:
        table[0, 0] += 1  # Beide korrekt
    elif A_pred == true_label and B_pred != true_label:
        table[0, 1] += 1  # Nur A korrekt
    elif A_pred != true_label and B_pred == true_label:
        table[1, 0] += 1  # Nur B korrekt
    else:
        table[1, 1] += 1  # Beide falsch
    
print("Kontingenztabelle:")
print(table)

# McNemar's Test durchführen
result = mcnemar(table, exact=True)
print(f"McNemar's Test P-Wert: {result.pvalue:.5f}")
