## Team: DTU-CSE Official
### Members:
- [Ashish](https://github.com/JustSurWHYving)
- [Raj Aryan](https://github.com/raj-aryan25)
- [Saksham Jain](https://github.com/sakshamjainwin)
- [Sparsh Jain](https://github.com/SparshJain769)

## Instructions to run the code
- Install the dependencies using `pip install -r requirements.txt`
- Specify the path to the test dataset in `evaluationApp.py`
- Run the code using `python evaluationApp.py`

## Description of the Approach

This project is centered around an intelligent bidding system for online advertisements that leverages a deep learning model and a series of rule-based adjustments to compute bid prices. The main components of the approach are:

1. **Click-Through Rate (CTR) Prediction with OuterPNN**  
   The core of the system is an Outer Product-based Neural Network (OPNN) implemented in `models/opnn.py`. The OPNN model:
   - Uses an embedding layer to encode sparse input features into dense latent representations.
   - Computes an outer product transformation of summed embeddings using a learnable kernel, capturing complex feature interactions.
   - Feeds the combined results into a Multi-Layer Perceptron (MLP) consisting of several fully connected layers with ReLU activations and dropout for regularization.
   - Outputs a probability through a final sigmoid activation function, representing the predicted CTR.

2. **Bid Price Calculation Strategy**  
   The bidding logic is implemented in `bid.py` and can be summarized as follows:
   - **Feature Extraction and Preprocessing:**  
     Bid request data is preprocessed by extracting both categorical and numerical features. Precomputed label encodings and mappings ensure that categorical features are efficiently handled.
   - **CTR and CVR Predictions:**  
     The model uses separate pre-trained CTR (OPNN) and CVR models for initial prediction. For certain advertisers (e.g., advertiser id "3358"), a more detailed strategy is applied by combining both CTR and CVR predictions to influence the final bid.
   - **Dynamic Bid Adjustment:**  
     A base bid is computed using the floor price and further adjusted by several multipliers:
       - **Advertiser-based Weighting:** Custom weights are applied depending on the advertiser’s identity.
       - **Ad Slot Characteristics:** Multipliers based on ad slot visibility, format, and the physical area of the ad are used.
       - **Random Variation:** A controlled random factor is introduced to avoid being too deterministic in the market.
   - **Final Decision Logic:**  
     The system applies a final constraint based on the predicted CTR. If the prediction meets certain criteria, the bid is adjusted with the calculated variation; otherwise, a fallback bid (often set to a default like 80) is used to ensure conservative bidding.

Overall, the approach combines deep learning-based CTR predictions with heuristic adjustments to optimize bid prices dynamically, ensuring that the bidding engine responds to both learned patterns in user behavior and the operational specifics of each bid request.

## Exploratory Data Analysis (EDA)
The dataset provided serves as a sample log of bid requests and associated metadata from the ad bidding system. Below is an outline of the EDA performed on this sample dataset:

### 1. Data Structure and Format
- **File Format:**  
  The file is a plain-text tab-separated values (TSV) file with no explicit header. Each line represents a single bid record.
  
- **Field Composition:**  
  On inspection, each row contains many fields representing different attributes such as:
  - A unique identifier (e.g., `82aed71bea7358c9a5be868deae30be0`), a bid ID.  
  - A timestamp-like field (e.g., `20130613000101373`), representing the request time.
  - A numerical flag (e.g., the value `1`), indicates the type of log entry: `1` (impression), `2` (click), `3` (conversion).
  - User agent strings (e.g., `"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.802.30 Safari/535.1 SE 2.X MetaSr 1.0"`) that contain browser and OS details.
  - IP address patterns (e.g., `60.187.41.*`) which appear to be partially masked.
  - Numerous numerical fields representing counts, measures, or flags (e.g., `94`, `100`, `1`, etc.). These correspond to attributes such as ad slot dimensions, floor price, click indicators, and other bidding-related metrics.

### 2. Distribution Analysis and Feature Insights
- **User Agent and Platform Distribution:**  
  By analyzing the user agent strings, one can derive insights into the platforms and browsers used by the audience (e.g., Chrome on Windows, Internet Explorer, etc.). This can help in understanding which segments drive more bid requests.

- **IP Address Patterns:**  
  The IP field formatted with wildcards suggests some level of aggregation – perhaps for privacy reasons. Analyzing these patterns may help segment data regionally.

- **Numerical Feature Distribution:**  
  Key numeric features (likely representing dimensions, floor prices, or click indicators) should be summarized using statistics such as mean, median, range, and standard deviation. Histograms for these values can uncover skewness, outliers, or clusters in the data.

- **Correlation Analysis:**  
  Evaluating the correlation between features (for example, between floor price and ad slot dimensions or click indicators) can provide insights into which variables are most predictive in the bidding process.

### 3. Visualizing the Data
- **Histograms and Box Plots:**  
  Plotting the distributions of key numeric features helps to visually inspect the spread and detect potential outliers.
  
- **Bar Charts for Categorical Features:**  
  Visualizations of frequencies for categorical data (e.g., browser types, ad exchanges) can highlight dominant groups.

- **Time-Series Analysis:**  
  If timestamps are parsed correctly, plotting bid request frequency over time can reveal temporal patterns (like peak ad activity).

## Feature Engineering
Carried out in the `utils/data.py` file (not provided in the code snippet).

### 1. Time Feature Transformation
- **Time Fraction Encoding:**  
  - The function `to_time_frac(hour, min, time_frac_dict)` divides each hour into four 15-minute segments.  
  - By iterating over the minute ranges defined in `time_frac_dict` for a specific hour, it converts the timestamp into a discrete "time fraction" index.  
  - **Purpose:** This transformation standardizes time information into 96 segments across a 24-hour period, simplifying temporal analysis.

### 2. Categorical Feature Indexing
- **Direct and Transformed Features:**  
  - The data is divided into several feature groups:
    - **First Group (`f1s`):** Includes features like weekday, hour, IP, region, city, ad exchange, domain, slot dimensions (id, width, height, visibility, format), creative, and advertiser.
    - **Second Group (`f1sp`):** Consists of features requiring additional processing like `useragent` and `slotprice`.
  - **Indexing Process:**
    - Each unique feature is assigned a unique index.
    - For each column, a default "other" category is created to handle unseen or rare feature values.
    - The feature index dictionary (`featindex`) is maintained to map these textual representations to their respective numerical indices.

### 3. Feature Transformations
- **User Agent Transformation:**  
  - The function `feat_trans(name, content)` processes the `useragent` field by:
    - Converting the string to lowercase.
    - Matching substrings to known operating systems (e.g., windows, ios, mac, android, linux) and browsers (e.g., chrome, firefox, safari).
    - Combining the detected OS and browser into a single standardized feature.
- **Slot Price Bucketing:**  
  - For the `slotprice` field, numeric values are bucketed into ranges:
    - Values greater than 100 become `"101+"`
    - Values between 51 and 100 become `"51-100"`
    - Values between 11 and 50 become `"11-50"`
    - Values between 1 and 10 become `"1-10"`
    - Zero remains `"0"`

### 4. Handling User Tags
- **User Tag Processing:**  
  - The function `getTags(content)` processes the `usertag` field:
    - If empty or newline, it defaults to `["null"]`.
    - Otherwise, the tags are split by commas and only the first five are retained.
  - A combined string of user tags is then used as a feature, which is also indexed.

### 5. LIBSVM Encoding
- **Data Conversion:**  
  - The `to_libsvm_encode()` function performs the complete encoding of the dataset:
    - It reads the `train.bid.csv` and `test.bid.csv` files.
    - For each line (representing an instance), it extracts features and converts them into their respective indices.
    - These indices are formatted as LIBSVM strings and written to output files (`train.bid.txt` and `test.bid.txt`).
  - **Feature Index Table:**  
    - A feature index mapping file (`feat.bid.txt`) is generated, listing all features and their corresponding indices.
    - This table is crucial for both the training of the CTR prediction model and ensuring consistency between training and testing data.

## Model Selection

### 1. Outer Product-based Neural Network (OPNN)
- **Architecture Overview:**  
  The OPNN model, defined in `models/opnn.py`, is specifically designed for capturing higher-order interactions among features. It achieves this by:
  - Using an embedding layer to convert sparse features into dense representations.
  - Applying an outer product transformation through a learnable kernel, which explicitly models pairwise interactions.
  - Passing the transformed features through a multi-layer perceptron (MLP) that consists of several fully connected layers with non-linear activations (ReLU) and dropout for regularization.
  - Producing a final CTR prediction by applying a sigmoid activation on the MLP’s output.

- **Key Hyperparameters and Design Choices:**  
  The model design involves careful selection of hyperparameters such as the latent dimension size, the number of features, and field counts. This allows the network to be tuned to the specifics of the ad dataset and the nature of feature interactions.
  
- **Rationale for Selection:**  
  The unique architecture of OPNN is chosen because it explicitly models multiplicative feature interactions, which are crucial in predicting user responses in online advertising. The additional non-linearity in the MLP further helps in learning complex patterns in high-dimensional data.

### 2. Integration of Multiple Predictive Models in the Bidding Process
- **CTR and CVR Models in Bid Calculation:**  
  In `bid.py`, the bidding logic integrates predictions from two distinct sources:
  - **CTR Model:**  
    The OPNN architecture is used as the primary model for CTR prediction. This prediction is critical in assessing the likelihood that a bid leads to user engagement.
  - **CVR Model:**  
    For certain advertisers (for example, advertiser id "3358"), a separate CVR model combo (XGB+DecisionTree Classifier) is also employed. This model delivers a conversion rate estimate that is then combined with the CTR prediction.

- **Model Loading and Deployment:**  
  The bid script dynamically loads pre-trained model parameters (saved in `.pth` files for the OPNN and using joblib for the other models). This allows for a flexible model deployment where models can be updated or replaced without changing the bidding logic.
  
- **Selection Strategy Rationale:**  
  The decision to use specialized models for CTR and CVR stems from the fact that different objectives (clicks vs. conversions) often require tailored modeling approaches. By isolating these components, the system can:
  - More accurately capture the distinct patterns in user behavior.
  - Adjust the bidding price not only based on the probability of a click but also considering downstream conversion events.
  
## Hyperparameter Tuning
In this project, several key hyperparameters were tuned manually to optimize the performance of the bidding system, particularly in the CTR prediction model through the OuterPNN architecture. The following outlines the aspects and approach to manual hyperparameter tuning:

1. **Embedding Dimensions and Latent Space:**
   - The hyperparameter `latent_dims` controls the size of the dense vector representations for each feature.
   - Manual trials were conducted by varying the latent dimensions to balance model expressiveness with overfitting risk. This choice was critical in ensuring that the model captures relevant user–feature interactions without excessive complexity.

2. **Network Architecture and Hidden Layer Sizes:**
   - The MLP portion of the OuterPNN model is constructed using a sequence of fully connected layers. In the implementation, a series of three hidden layers is used with neuron sizes defined as `[300, 300, 300]`.
   - The team manually tested with different numbers and sizes of hidden layers, adjusting the number of neurons to improve learning capacity while keeping computational requirements in check.

3. **Dropout and Regularization:**
   - Dropout is applied after each activation (with a dropout probability set at 0.2) to prevent overfitting.
   - The dropout probability was manually adjusted through trial and error, assessing its influence on validation performance. Fine-tuning these regularization parameters helped ensure that the model generalized well to unseen bid requests.

4. **Learning Rate and Weight Initialization:**
   - While not explicitly detailed in the provided code snippets, learning rates and weight initialization methods (e.g., Xavier uniform initialization and uniform distributions for the linear layers’ weights) were also subject to manual tuning.

## Evaluation Strategy
The evaluation strategy in this project was carried out by splitting the training data into separate train and test sets. This train-test split enabled the team to validate the bidding system’s performance and tune the models.

# Validation Results
- CTR Model Validation AUC: 0.9094
- CVR Model Validation AUC: 0.8889
