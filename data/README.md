@"
# Data Sources

Download these datasets and place them in this folder.

## Nigeria datasets (Electric Sheep Africa, Hugging Face)

- Point of Sale Records — Nigeria (800,000 rows):
  https://huggingface.co/datasets/electricsheepafrica/africa-synth-retail-and-ecommerce-point-of-sale-records-nigeria

- Supply Chain Logistics Data — Nigeria (400,000 rows):
  https://huggingface.co/datasets/electricsheepafrica/africa-synth-retail-and-ecommerce-supply-chain-logistics-data-nigeria

- Last Mile Delivery Data — Nigeria (150,000 rows):
  https://huggingface.co/datasets/electricsheepafrica/africa-synth-retail-and-ecommerce-last-mile-delivery-data-nigeria

## Routing benchmark (Kigali, Zenodo)

- Kigali E-Commerce Delivery Network Dataset (Niyibizi, 2026):
  https://doi.org/10.5281/zenodo.20151874

## Note

The Electric Sheep Africa datasets are synthetic, parameterised from published Sub-Saharan African literature. They are used for simulation and framework development.
"@ | Out-File -FilePath data\README.md -Encoding utf8