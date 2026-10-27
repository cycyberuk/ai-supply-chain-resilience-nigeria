@"
# AI-Driven Supply Chain Resilience in Nigerian E-Commerce

Reproducible simulation framework for demand sensing, inventory optimisation, and last-mile logistics in Nigerian e-commerce.

## Repository structure

- `code/` — Python scripts for the three modules
- `data/` — download instructions (datasets not committed)
- `outputs/` — CSV results and figures
- `paper/` — manuscript (added separately)

## Setup

pip install -r requirements.txt

Download the datasets (see `data/README.md`) and place them in `data/`.

## Reproduce results

python code/demand_sensing.py
python code/inventory_optimization.py
python code/module3a_eta_prediction.py
python code/module3b_vrp_routing.py
"@ | Out-File -FilePath README.md -Encoding utf8