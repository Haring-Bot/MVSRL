# Assuming b and c are the discordant counts from your contingency table
# b = (Model 1 Correct, Model 2 Incorrect)
# c = (Model 1 Incorrect, Model 2 Correct)
mcnemar_statistic = ((abs(b) - abs(c))**2) / (abs(b) + abs(c))

# Degrees of freedom is always 1 for McNemar's test
df = 1

# Use chi-square distribution to find p-value (assuming significance level alpha = 0.05)
from scipy.stats import chi2
p_value = chi2.sf(mcnemar_statistic, df)
