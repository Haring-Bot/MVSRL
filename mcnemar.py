from statsmodels.stats.contingency_tables import mcnemar

# Kontingenztabelle erstellen
table = np.zeros((2, 2))

# Vergleiche die Vorhersagen für PCA und als Alternative SVD
for pca_pred, svd_pred, true_label in zip(y_pred_pca, y_pred_svd, y_test):
    if pca_pred == true_label and svd_pred == true_label:
        table[0, 0] += 1  # Beide korrekt
    elif pca_pred == true_label and svd_pred != true_label:
        table[0, 1] += 1  # Nur PCA korrekt
    elif pca_pred != true_label and svd_pred == true_label:
        table[1, 0] += 1  # Nur SVD korrekt
    else:
        table[1, 1] += 1  # Beide falsch

# McNemar's Test durchführen
result = mcnemar(table, exact=True)
print(f"McNemar's Test P-Wert: {result.pvalue:.5f}")
