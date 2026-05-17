
# Calibration Analysis

## Reliability diagram interpretation

The saved reliability diagram (`figures/reliability-diagram.png`) reveals a significant disparity between the model's predicted confidence and its actual empirical accuracy. Specifically, the model displays extreme **over-confidence**. For instance, in the high-confidence buckets (around the 0.8–0.9 range), one prediction reached a confidence of **0.8979**, yet the accuracy for that bucket was **0.0**. The bars representing empirical accuracy remain far below the "Perfect Calibration" diagonal line across all populated bins, indicating that the model's internal probability scores are much higher than its actual success rate.

## Expected Calibration Error

The reported **Expected Calibration Error (ECE) is 0.6192**. This high value indicates that the model is poorly calibrated and essentially untrustworthy for direct production use. An ECE of ~62% means there is a massive gap between what the model "thinks" it knows and what it actually predicts correctly. In a production environment, relying on these raw probability scores for decision-making or automated workflows would be highly risky.

## A specific calibration pattern

A distinct pattern of **over-confidence in incorrect classifications** is evident. The model predicted several classes with mid-to-high confidence (e.g., 0.52 and 0.89) while failing to achieve a single correct prediction in the sample set, resulting in an accuracy of **0.0**. This likely arose because the model (DistilBERT) was fine-tuned on a specific dataset where it learned to minimize Loss by pushing Logits to extremes, leading to "over-fit" probability distributions that do not reflect true empirical certainty.

## A proposed engineering action

Based on these findings, I propose implementing **Temperature Scaling** as a primary engineering action. By applying a temperature parameter ($T > 1$) to the logits before the softmax layer, we can "soften" the probability distribution, effectively pushing the over-confident scores down to more realistic levels. Additionally, we should implement **threshold-based abstention**, where the system refuses to provide a label if the model's confidence hasn't been properly calibrated or if the ECE remains above a certain threshold for specific buckets.