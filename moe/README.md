# Image MoE Detector

Architecture

Image
↓

Gating Network

↓

Gate Values

×

Expert Scores (CSV)

↓

MLP Combiner

↓

Probability(Fake)

---

CSV format

image,label,expert1,expert2,...

Example

cat.png,0,0.11,0.45,0.81,0.67

dog.png,1,0.98,0.87,0.94,0.91