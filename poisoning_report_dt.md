
# Data Poisoning Performance Degradation Report (Decision Tree)

This report shows how a Decision Tree model's performance metrics degrade as the percentage of flipped labels (poison) in the training data increases.

## Summary Table

| poison_level   |   accuracy |   precision |    recall |        f1 |      auc |
|:---------------|-----------:|------------:|----------:|----------:|---------:|
| 2%             |   0.952506 |   0.0725429 | 0.0996785 | 0.0839729 | 0.53558  |
| 8%             |   0.837504 |   0.089307  | 0.107358  | 0.0975039 | 0.50478  |
| 15%            |   0.725241 |   0.152529  | 0.179201  | 0.164793  | 0.501057 |
| 30%            |   0.571241 |   0.299125  | 0.316604  | 0.307617  | 0.498506 |

## Analysis

As the poisoning level increases, a clear degradation is observed across all key metrics. Decision Trees can be particularly sensitive to label noise as incorrect labels can easily misguide the splitting criteria at each node, leading to a poorly structured tree.
