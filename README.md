# LaptopPricePrediction
Predicting laptop prices using machine learning. This project leverages a dataset containing various features such as manufacturer, processor type, RAM, storage capacity, and more to train a machine learning model capable of predicting laptop prices. The process involves thorough data exploration, preprocessing, visualization, and the application of a Random Forest Classifier.

## Key Steps:
## Data Exploration
Load and explore the dataset to understand its structure.
Check for missing values and analyze basic statistics.
## Data Preprocessing
Encode categorical variables using Label Encoding.
Assess multicollinearity through Variance Inflation Factor (VIF) calculations.
Drop unnecessary columns to enhance model efficiency.
            ```from sklearn import preprocessing
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                   label_encoder = preprocessing.LabelEncoder()
                    df['Manufacturer'] = label_encoder.fit_transform(df['Manufacturer'])
                    variables = df[['Manufacturer', 'IntelCore(i-)', 'IntelCoreGen', 'processing speed(GHz)', 'Ram(gb)', 'HDD(gb)', 'SSD(gb)', 'Graphics(gb)', 'ScreenSize(inch)']]
### Calculate variance inflation factor (VIF)
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif['Features'] = variables.columns
    
    ### Drop unnecessary columns
    df = df.drop(['IntelCore(i-)', 'IntelCoreGen', 'processing speed(GHz)', 'HDD(gb)', 'ScreenSize(inch)'], axis=1)
    
    ## Data Visualization
    Create visualizations to understand relationships between variables.
    Utilize pair plots and heatmaps for better insights.
    ```import seaborn as sb
    import matplotlib.pyplot as plt
    
    ### Pairplot for visualization
    sb.pairplot(df)
    
    ### Heatmap for correlation
    corr = df.corr(method='pearson')
    sb.heatmap(corr, annot=True, cmap=plt.cm.CMRmap_r)
    
    ## Data Splitting
    Split the dataset into training and testing sets for model evaluation.
    ```from sklearn.model_selection import train_test_split
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ## Choosing & Training Model
    Implement a Random Forest Classifier for predicting laptop prices.
    Evaluate the model's performance using R2 Score.
    ```from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import r2_score
    
    ### Initialize and train the Random Forest model
    model_RF = RandomForestClassifier(random_state=0)
    model_RF.fit(X, y)
    
    ### Predict on the test set
    y_pred = model_RF.predict(X_test)
    
    ### Evaluate the model
    print(f"R2_Score: {r2_score(y_test, y_pred)*100}%")
    print("Predicted Values:")
    print(y_pred[:10])
    
    ## Results
    Achieved an impressive accuracy of 99.0908% in predicting laptop prices.
