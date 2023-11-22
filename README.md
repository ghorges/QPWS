# QPWS(Qualitative Percentile Weighted Sampling)

qpws is a versatile Python program designed for performing feature selection on near-infrared spectroscopy data. Feature selection is a critical step in data analysis, and it allows you to identify and retain the most informative features or variables from your data. The goal of qpws is to help researchers, analysts, and scientists streamline the process of selecting the most relevant features from their near-infrared spectroscopy datasets.

## Overview

Near-infrared spectroscopy is a powerful analytical technique used in various fields, including chemistry, pharmaceuticals, food science, and more. In qualitative analysis, it's essential to extract and focus on the most relevant variables to draw meaningful insights from the data.

qpws offers a method of feature selection known as Qualitative Percentile Weighted Sampling. This method helps identify and retain the most informative variables from a dataset, reducing data dimensionality and improving the quality of qualitative analysis.

## Features

- **Tailored for Qualitative Analysis:** qpws is specifically designed to excel in qualitative analysis. Its feature selection methodology is optimized to enhance the quality of qualitative insights, making it superior to other generic feature selection methods.
- **Parallel Processing:** qpws leverages multi-threading to accelerate the feature selection process. This means you can efficiently analyze large datasets and select the most relevant features in a shorter amount of time.
- **Cross Validation (CV):** The program includes a CV module, which enables you to validate the selected features' impact on the model's performance. This can be vital for ensuring the quality of your feature selection.
- **Visualization:** qpws provides visualizations to help you understand the feature selection process and the impact of different variables on model accuracy. The program generates 3D plots and other visualizations to assist in your analysis.

## How to Use qpws

1. **Install Dependencies:** Before using qpws, make sure to install the necessary Python libraries, including NumPy, scikit-learn, pandas, and Matplotlib.
2. **Import Your Data:** Prepare your near-infrared spectroscopy data in a suitable format and import it into the program.
3. **Adjust Parameters:** Modify the program's parameters to fit your specific dataset and requirements. This may include setting the number of components, threads, and other options.
4. **Run Feature Selection:** Execute the qpws algorithm to perform feature selection. The program will run parallel threads to efficiently analyze the data.
5. **Evaluate Results:** qpws will provide you with the best-selected features, their impact on model accuracy, and other relevant information. You can use these results for subsequent data analysis or modeling.
6. **Visualize the Process:** The program generates visualizations, including 3D plots, to help you understand the feature selection process and its outcomes.

## Example Usage

Here's a basic example of how to use qpws in your Python code:

```
pythonCopy code
import qpws

# Load and prepare your near-infrared spectroscopy data
x, y = prepare_data()

# Run the feature selection algorithm
qpws.Algorithm(x, y)
```

## Performance Testing

### Data Source
The performance testing was conducted using the dataset provided in the following research paper:
- **Paper:** [Link to Paper](https://doi.org/10.1038/s41598-022-17810-y)
- **Open Dataset:** [Download Dataset](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-022-17810-y/MediaObjects/41598_2022_17810_MOESM1_ESM.zip)

### Test Configuration

| Feature Selection | Band Selection | Processing Speed | Method   | Accuracy |
|-------------------|-----------------|-------------------|----------|----------|
| QPWS              | 2               | 20ms              | PLS-DA   | 100%     |

### Comparison with Research Paper

| Paper Feature Selection | Paper Band Selection | Paper Processing Speed | Paper Method | Paper Accuracy |
|-------------------------|----------------------|--------------|-------------------------|----------------|
| PCA                     | 6                    | 30ms          | EML                    | 100%           |

The performance testing results indicate that the use of QPWS for feature selection, with only 2 selected bands, outperformed the approach suggested in the original research paper that employed PCA with 6 selected bands. The QPWS method not only demonstrated a more efficient selection of spectral bands but also yielded superior results, achieving 100% accuracy at a processing speed of 20ms. This contrasts with the paper's approach using PCA, which, despite selecting a larger number of bands (6), achieved the same accuracy but at a slightly slower processing speed (30ms).

The findings suggest that QPWS is a more effective feature selection method for the given dataset, providing a streamlined approach with improved performance metrics. This highlights the importance of carefully selecting and evaluating feature extraction techniques in spectral analysis applications.



## License

qpws is open-source software released under the MIT License. You are free to use and modify it for your specific needs. See the [LICENSE](https://github.com/ghorges/QPWS/blob/main/LICENSE) file for more details.

## Contributing

We welcome contributions from the community. If you have suggestions for improving qpws, please submit a pull request or open an issue on our [GitHub repository](https://github.com/ghorges/qpws).

## Contact

If you have any questions or need support, please contact us at [1298394633@qq.com](mailto:1298394633@qq.com).
