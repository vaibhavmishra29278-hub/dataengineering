PRACTICAL1 DATA EXTRACTION USING WEB SCRAPING
import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://www.worldometers.info/world-population/population-by-country/"
src = requests.get(url)

print("Status code:", src.status_code)

soup = BeautifulSoup(src.content, "lxml")

table = soup.find("table")

# Extract headers
cols = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

# Extract rows
rows = table.find("tbody").find_all("tr")

data = []
for r in rows:
    row = [td.get_text(strip=True) for td in r.find_all("td")]
    data.append(row)

# Create dataframe
df = pd.DataFrame(data, columns=cols)

print(df.head())

PRACTICAL 2 CREATE ML PIPELING USING PYTHON LIBRARIES
# Import libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["class"] = data.target

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Create pipelines for models
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("model", LogisticRegression())
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("model", DecisionTreeClassifier())
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("model", GaussianNB())
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("model", RandomForestClassifier())
    ])
}


# Train models and check accuracy
for name, model in pipelines.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(name, "Accuracy:", acc)

    PRACTICAL 7A SIMPLE PYSPARK DRIVER PROGRAM
    # Import libraries
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("SquareNumbers").getOrCreate()

# Get Spark context
sc = spark.sparkContext

# Create RDD
numbers = sc.parallelize([1, 2, 3, 4, 5])

# Square each number
squared_numbers = numbers.map(lambda x: x * x)

# Show result
print(squared_numbers.collect())

# Stop Spark
spark.stop()

PRACTICAL 7B WORKING WITH DIFF TRANSFORMATION AND ACTION BY RDD BY 
from pyspark import SparkContext

sc = SparkContext("local","Iris")

iris = sc.textFile("iris.csv")
print(iris.collect())

# Partition info
for i,p in enumerate(iris.glom().collect()):
    print("Partition",i+1,":",len(p),"records")

# Repartition
iris = iris.repartition(8)
print("Total Partitions:",iris.getNumPartitions())

# Remove header
header = iris.first()
data = iris.filter(lambda x: x!=header)

# Map and flatMap
print(data.map(lambda x:x.split(",")).collect())
flat = iris.flatMap(lambda x:x.split(","))
print(flat.take(5))

# Filter Virginica
print(iris.filter(lambda x:"virginica" in x.lower()).map(lambda x:(x,1)).collect())

# Species count
species = ['setosa','versicolor','virginica']
count = iris.flatMap(lambda x:x.split(",")) \
            .filter(lambda x:x.lower() in species) \
            .map(lambda x:(x,1)) \
            .reduceByKey(lambda a,b:a+b)

print(count.collect())

sc.stop()

PRACTICAL 8 DATA EXPLORATION USING SPARK SQL
from pyspark import SparkContext

sc = SparkContext("local","Iris")

iris = sc.textFile("iris.csv")
print(iris.collect())

# Partition info
for i,p in enumerate(iris.glom().collect()):
    print("Partition",i+1,":",len(p),"records")

# Repartition
iris = iris.repartition(8)
print("Total Partitions:",iris.getNumPartitions())

# Remove header
header = iris.first()
data = iris.filter(lambda x: x!=header)

# Map and flatMap
print(data.map(lambda x:x.split(",")).collect())
flat = iris.flatMap(lambda x:x.split(","))
print(flat.take(5))

# Filter Virginica
print(iris.filter(lambda x:"virginica" in x.lower()).map(lambda x:(x,1)).collect())

# Species count
species = ['setosa','versicolor','virginica']
count = iris.flatMap(lambda x:x.split(",")) \
            .filter(lambda x:x.lower() in species) \
            .map(lambda x:(x,1)) \
            .reduceByKey(lambda a,b:a+b)

print(count.collect())

sc.stop()

PRACTICAL 9 CREATE A PYSPARK MLLIB
from pyspark import SparkContext

sc = SparkContext("local","Iris")

iris = sc.textFile("iris.csv")
print(iris.collect())

# Partition info
for i,p in enumerate(iris.glom().collect()):
    print("Partition",i+1,":",len(p),"records")

# Repartition
iris = iris.repartition(8)
print("Total Partitions:",iris.getNumPartitions())

# Remove header
header = iris.first()
data = iris.filter(lambda x: x!=header)

# Map and flatMap
print(data.map(lambda x:x.split(",")).collect())
flat = iris.flatMap(lambda x:x.split(","))
print(flat.take(5))

# Filter Virginica
print(iris.filter(lambda x:"virginica" in x.lower()).map(lambda x:(x,1)).collect())

# Species count
species = ['setosa','versicolor','virginica']
count = iris.flatMap(lambda x:x.split(",")) \
            .filter(lambda x:x.lower() in species) \
            .map(lambda x:(x,1)) \
            .reduceByKey(lambda a,b:a+b)

print(count.collect())

sc.stop()
