# Important note for later
- Limitations:
It is impossible to train the model to recognize and assign some DDI because of lack of data. These interactions, quantified as those with <100 D-D samples, are grouped together and labeled as 'Other' by the model. Unfortunately, many of these interactions are severe. So as always, trust a educated professional. Ask your pharmacist. 
![alt text](image.png) -- INSERT IMAGE OF THE CHART HERE


## Other things to explain along the way:
- F1 score and why
- Different drug interactions and why
- Be able to explain all the choies, XGBoost, random tree, etc.
- Confusion matrices and where to see the terrible one
- Having the per-prediction be limited to the predicted interaction, not all interaction types. (speed and for experiment sake(only consider why the model assigned a specific interaction, not ALL interactions))
- Model, features, predict, and explain stored within separate .py files.  
- MIT thing for reproducability/usage
- headless thing for streamlit



## Features to make sure to implement
- Dictionary converting the numerical interaaction type into the actual type
- Dictionary labeling what molecular structure is assigned to each feature. 