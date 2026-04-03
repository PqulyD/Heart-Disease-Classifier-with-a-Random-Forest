# Heart-Disease-Classifier-with-a-Random-Forest
This is my first machine learning project that does not involve using neural networks. The point of this project is to dip my toes in other machine learning models so that I can extend my ML knowledge. This project uses a random forest to teach a program to classify if a patient has heart disease or not. The data that was used was found on KAGGLE where it is aggregate data across many hospitals. 
To test my undertanding on how random forests work, I have both a basic random forest and a random forest with hyperparameters. I have this so I can test if my knowledge on how the hyperparameters work can allow me to get a higher accuracy than a basic random forest. As you can see from this: 

rfc accuracy:        0.907608695652174

rfc2 accuracy:       0.9239130434782609

Difference:          0.016304347826086918

By using hyperparameters: 

rfc2 = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=12,

    min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=67, n_jobs=-1


I was able to achieve a higher accuray by the difference above. 
The model consists of 1,000 trees (n_estimators = 1000); the average of more trees will provide a more accurate prediction. Each tree split is based on entropy (criterion = 'entropy'); entropy provides a measure of information gain and creates purer splits. Therefore, it is a good criterion to use for medical classification problem datasets. Each tree is grown to a maximum depth (max_depth = 12); this depth is good to find patterns within the dataset but is also shallow enough to prevent overfitting (when the model learns the noise within the dataset instead of learning the generalizations of the signal within the dataset). A node must have a minimum of 5 samples to perform a split (min_samples_split = 5); additionally, each resulting leaf must have a minimum of 2 samples (min_samples_leaf = 2) to ensure that the model does not make a prediction based on one patient's record only (this would be memorization, not learning). For each split, the maximum number of features to be used for any particular split is based on the square root of the total number of features in the dataset (max_features = 'sqrt'). This random selection of features for each split is what makes a Random Forest model "random"; if this randomness were to be removed, all trees would be similar or identical, thereby decreasing the effect of the ensemble model. Finally, random_state = 67 ensures that the random number generator is seeded to guarantee the reproducibility of the results from one run to another, and n_jobs = -1 enables the use of all available CPU cores for training, resulting in faster training times without affecting the model accuracy.

