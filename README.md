# 🏁F1 Winner Predictor (2026 Season Edition)

An interactive Machine Learning dashboard built with **Python**, **Scikit-Learn**, and **Streamlit** that predicts Formula 1 race outcomes. This model is specifically tuned for the **2026 Technical Regulations**, accounting for the 50/50 power unit split and active aerodynamics.

## Features
* **Real-Time Data:** Fetches live results from the 2024, 2025, and 2026 seasons using the `FastF1` API.
* **Weighted Intelligence:** Uses a **Random Forest Regressor** that gives 3x more weight to 2026 "New Era" races to ensure accuracy in the current power struggle between Mercedes and Ferrari.
* **Interactive UI:** A Streamlit-based dashboard where users can input qualifying grids and see immediate podium predictions.

## How it Works
The model analyzes three primary features to determine the finishing position:
1. **Grid Position:** Historically the strongest predictor of success.
2. **Team ID:** Captures constructor performance (Crucial for the 2026 Mercedes dominance).
3. **Driver ID:** Factors in individual driver skill and consistency.



## Getting Started

### Prerequisites
* Python 3.9+
* A stable internet connection (to fetch FastF1 data)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/priya-gurung/F1-Winner-Predictor.git
   cd F1-Winner-Predictor
