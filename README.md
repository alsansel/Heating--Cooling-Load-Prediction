# Heating-Cooling-Load-Prediction

This repository contains the full pipeline for building, evaluating, and deploying a heating  & cooling load prediction model using EDA, linear regression, random forest, and SHAP explainability. Dataset is from the UCI Energy Efficiency dataset and contains eight architectural parameters.

| Variable             | Description                      | Type                  |
| -------------------- | -------------------------------- | --------------------- |
| Relative Compactness | Building surface-to-volume ratio | Numerical             |
| Surface Area         | Facade area \[m²]                | Numerical             |
| Wall Area            | Wall area \[m²]                  | Numerical             |
| Roof Area            | Roof area \[m²]                  | Numerical             |
| Overall Height       | Building height \[m]             | Numerical             |
| Orientation          | 2 = N, 3 = W, 4 = E, 5 = S       | Categorical → One Hot |
| Glazing Area         | Percentage of glazing \[%]       | Numerical             |
| Glazing Area Dist.   | Type of glazing distribution     | Categorical → One Hot |
| Y1 Heating Load      | Heating energy demand \[kWh]     | Target                |
| Y2 Cooling Load      | Cooling energy demand \[kWh]     | Target                |

