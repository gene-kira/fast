import os
import time
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("CybersecurityAI").getOrCreate()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "benign_files_path": "path/to/benign/files",
    "malicious_files_path": "path/to/malicious/files",
    "phishing_emails_csv": "path/to/phishing_emails.csv",
    "normal_traffic_csv": "path/to/normal_traffic.csv",
    "model_save_path": "models"
}

def load_data(spark, benign_path, malicious_path):
    benign_df = spark.read.format("text").load(benign_path)
    benign_df = benign_df.withColumnRenamed("value", "content").withColumn("label", lit(0))
    
    malicious_df = spark.read.format("text").load(malicious_path)
    malicious_df = malicious_df.withColumnRenamed("value", "content").withColumn("label", lit(1))
    
    return benign_df.union(malicious_df)

def load_email_data(spark, phishing_csv, normal_csv):
    phishing_df = spark.read.csv(phishing_csv, header=True, inferSchema=True)
    phishing_df = phishing_df.withColumn("label", lit(1))
    
    normal_df = spark.read.csv(normal_csv, header=True, inferSchema=True)
    normal_df = normal_df.withColumn("label", lit(0))
    
    return phishing_df.union(normal_df)

def load_traffic_data(spark, normal_csv):
    traffic_df = spark.read.csv(normal_csv, header=True, inferSchema=True)
    traffic_df = traffic_df.withColumnRenamed("content", "traffic_content").withColumn("label", lit(0))
    return traffic_df

def preprocess_text(df, text_col="content"):
    hashingTF = HashingTF(inputCol=text_col, outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    
    featurized_data = hashingTF.transform(df)
    rescaled_data = idf.fit(featurized_data).transform(featurized_data)
    return rescaled_data

def preprocess_traffic(df):
    assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    
    assembled_df = assembler.transform(df)
    scaled_df = scaler.fit(assembled_df).transform(assembled_df)
    return scaled_df

def train_model(df, model_type='rf'):
    if model_type == 'rf':
        classifier = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures")
    elif model_type == 'gbt':
        classifier = GBTClassifier(labelCol="label", featuresCol="scaledFeatures")
    else:
        logging.error(f"Unknown model type: {model_type}")
        return None
    
    paramGrid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, [2, 4, 6]) \
        .addGrid(classifier.numTrees, [10, 50, 100]) \
        .build()
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
    
    crossval = CrossValidator(estimator=classifier,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)
    
    cvModel = crossval.fit(df)
    best_model = cvModel.bestModel
    return best_model

def evaluate_model(model, df):
    predictions = model.transform(df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
    auc = evaluator.evaluate(predictions)
    logging.info(f"AUC: {auc}")
    
    # Convert to Pandas for detailed report
    pd_predictions = predictions.select("label", "prediction").toPandas()
    f1_score = np.round(pd_predictions[pd_predictions['label'] == pd_predictions['prediction']].shape[0] / pd_predictions.shape[0], 2)
    logging.info(f"F1 Score: {f1_score}")
    
    return auc, f1_score

def save_model(model, filename):
    try:
        model.write().overwrite().save(os.path.join(CONFIG["model_save_path"], filename))
        logging.info(f"Model saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(filename):
    try:
        model = spark.read().load(os.path.join(CONFIG["model_save_path"], filename))
        logging.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def display_alerts(alerts):
    if alerts:
        logging.warning("Alerts detected:")
        for alert in alerts:
            logging.warning(f"- {alert}")
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(alerts)), [1] * len(alerts), tick_label=alerts)
        plt.title('Detected Alerts')
        plt.xlabel('Alerts')
        plt.ylabel('Count')
        plt.show()
    else:
        logging.info("No alerts detected.")

def real_time_monitoring(model_malware, model_phishing, model_intrusion):
    while True:
        try:
            # Collect new data (simulated here)
            benign_samples = load_data(spark, CONFIG["benign_files_path"], CONFIG["malicious_files_path"])
            email_data = load_email_data(spark, CONFIG["phishing_emails_csv"], CONFIG["normal_traffic_csv"])
            traffic_data = load_traffic_data(spark, CONFIG["normal_traffic_csv"])

            # Preprocess data
            malware_features = preprocess_text(benign_samples)
            email_features = preprocess_text(email_data)
            traffic_features = preprocess_traffic(traffic_data)

            # Split data into training and test sets
            train_malware, test_malware = malware_features.randomSplit([0.8, 0.2], seed=42)
            train_email, test_email = email_features.randomSplit([0.8, 0.2], seed=42)
            train_traffic, test_traffic = traffic_features.randomSplit([0.8, 0.2], seed=42)

            # Evaluate models
            evaluate_model(model_malware, test_malware)
            evaluate_model(model_phishing, test_email)
            evaluate_model(model_intrusion, test_traffic)

            # Make predictions and generate alerts
            malware_predictions = model_malware.transform(test_malware).select("prediction").toPandas()
            email_predictions = model_phishing.transform(test_email).select("prediction").toPandas()
            traffic_predictions = model_intrusion.transform(test_traffic).select("prediction").toPandas()

            alerts = []
            if any(malware_predictions['prediction'] == 1):
                alerts.append("Suspicious malware detected")
            if any(email_predictions['prediction'] == 1):
                alerts.append("Phishing attempt detected")
            if any(traffic_predictions['prediction'] == 1):
                alerts.append("Suspicious network traffic detected")

            display_alerts(alerts)

        except Exception as e:
            logging.error(f"Error during real-time monitoring: {e}")

        # Sleep for a while before the next iteration
        time.sleep(3600)  # Check every hour

def main():
    try:
        # Collect and load data
        benign_samples = load_data(spark, CONFIG["benign_files_path"], CONFIG["malicious_files_path"])
        email_data = load_email_data(spark, CONFIG["phishing_emails_csv"], CONFIG["normal_traffic_csv"])
        traffic_data = load_traffic_data(spark, CONFIG["normal_traffic_csv"])

        # Preprocess data
        malware_features = preprocess_text(benign_samples)
        email_features = preprocess_text(email_data)
        traffic_features = preprocess_traffic(traffic_data)

        # Split data into training and test sets
        train_malware, test_malware = malware_features.randomSplit([0.8, 0.2], seed=42)
        train_email, test_email = email_features.randomSplit([0.8, 0.2], seed=42)
        train_traffic, test_traffic = traffic_features.randomSplit([0.8, 0.2], seed=42)

        # Train models
        model_malware = train_model(train_malware, model_type='rf')
        model_phishing = train_model(train_email, model_type='rf')
        model_intrusion = train_model(train_traffic, model_type='gbt')

        if model_malware is None or model_phishing is None or model_intrusion is None:
            logging.error("Model training failed. Exiting...")
            return

        # Evaluate models
        evaluate_model(model_malware, test_malware)
        evaluate_model(model_phishing, test_email)
        evaluate_model(model_intrusion, test_traffic)

        # Save models
        save_model(model_malware, 'malware_detection_model')
        save_model(model_phishing, 'phishing_detection_model')
        save_model(model_intrusion, 'traffic_detection_model')

        # Start real-time monitoring
        real_time_monitoring(model_malware, model_phishing, model_intrusion)

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
