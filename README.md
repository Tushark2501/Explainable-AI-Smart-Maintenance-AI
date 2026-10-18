# Smart Maintenance AI

Explainable multimodal predictive maintenance: enter machine readings to get a health report, then browse every result and graph from the notebook.

## Machine readings

Start from a case run in the notebook, or adjust any value.

Product type

## Machine health report

0%

failure probability

## Live graphs

These redraw every time you move a reading above.

### Operating map

Your machine (dot) against the power and overstrain limits that cause failures.

### Distance from normal

How far each reading sits from the dataset average, in standard deviations.

### What-if sweeps

Failure probability if one reading changed while the others stay as set. The dot is where you are now.

## Multimodal fusion

Combines sensor risk, remaining useful life, a surface image and a maintenance log using the weights from the notebook (35% / 30% / 20% / 15%).

[x] Follow the machine readings above. Remaining life, image defect and log are filled in from your inputs as an illustration. Change any control below to take over.

Sensor failure probability (%)

Remaining useful life (cycles)

Surface image defect

Image confidence (%)

Maintenance log

## Combined assessment

overall risk

Share of overall risk: sensor (teal), remaining life (amber), image (red), log (grey). Status changes at 25%, 50% and 75%.

### Attention map

### Remaining life

### Recommendations

## Notebook results

Every graph below was produced in the Colab notebook. Select a graph to enlarge it.

**10,000**AI4I machine records, 339 failures (3.39%)

**0.996**XGBoost ROC-AUC

**25.1 cycles**LSTM remaining-life MAE (RMSE 38.8)

**91%**Steel-defect CNN accuracy, 6 classes

Built from Smart_Maintenance_AI.ipynb. Datasets: AI4I 2020, NASA C-MAPSS FD001, NEU surface defects. Prediction rows labelled "Notebook output" are the values printed in the notebook; the rest are in-browser estimates.
