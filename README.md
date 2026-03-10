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

Practical No. 3: Working with NoSQL database – hbase.
$ hbase shell
>create ‘studentrj’,’studid’,’studname’,’studaddr’
>list
>describe ’studentrj’
>put ‘studentrj’,’1’,’studid:id’,101
>put ‘studentrj’,1’,’studname:name’,’Kartik’
>put ‘studentrj’,’1’’studaddr:addr’,’Mumbai’
>scan ‘studentrj
Practical No. 4: Simulating Datawarehouse environment.
$hive
>create database college;
>use college;
>create table student(rollno int,name string,course string,marks float) row format delimited fields terminated by ’,’ stored as textfile;
>exit
  >;
>gedit studdata.txt
Inside enter:
101,Kartik,DS,35
102,Shruti,CS,45
103,Samruddhi,IT,35
Contorl+S and close it
$hive
>use college;
>describe formatted student;
>load data local inpath ‘/home/cloudera/studdata.txt’ into table student;
>select*from student;

Practical No 5 Working with different types of tables in hive
Internal Table : 
$hive 
> create database collage1; 
> show databases; 
> use collage1;
> exit;
$gedit studdata.txt
---TXT.FILE DATA---
 101,AAA,CS,75
 102,BBB,DS,67
 103,CCC,UT,87
$hive
> use college1;
> describe formatted student;
> load data local inpath '/home/coludera/studdata.csv' into table student;
>select * form student;
> create table customer (id int , 
name string, 
dob string,
email string,
contact string,
add string,
gender string)
row format delimited fields terminated by ‘,’ ;
> exit;
$ gedit cusdata.csv;
$ hive;
$ hdfs dfs -mkdir /hadoop
$ hdfs dfs -mkdir /hadoop/data
$ hdfs dfs -put custdata.csv /hadoop/data
$ hive
> use licdw;
> load data inpath '/hadoop/data/custdata.csv'into table customer;
> select* from customer;

External
>use licdw;
>create external table policy_details (id int , 
name string, 
type string, 
age_criteria int, 
tenure int , 
maturity string) row format delimited fields terminated by ‘,’ 
stored as textfile location ‘/home/cloudera/policydetail_data’ ; 
>load data local inpath ‘/home/cloudera/Desktop/policydetails.csv’ into table policy_details; 
> select * from policy_details ;

Temporary Table
>create temporary table policy_details_temp ( id int,
name string,
type string,
age_criteria int ,
tenure int,
maturity string);
> describe policy_details_temp ;
> insert into policy_details_temp values (32, ‘abc’ , ‘h’ , 32 , 3 , ‘40%’);
>insert into policy_details_temp values (33, ‘afc’ , ‘r’ , 32 , 3 , ‘44%’);
>show tables 
>select * from policy_det_temp;
>create table policy_det_dup as select * from policy_deatils;
>select * from plicy_det_dup;
>create table policy_det_like like policy_details;
>select * from policy_det_like;

Practical 5 : B. Demonstrating table partitioning , clustering (Bucketing in hive)
1. Create and partition a table named emppartition on the mobile no column and load data into it
$hive
>create database licdw;
>use licdw;
>create table emppartition(
empid int,
name string
)
partitioned by (mno string)
row format delimited
fields terminated by ",";

>set hive.exec.dynamic.partition.mode=nonstrict;
>insert into emppartition partition(mno) values(1,'uday','89272861');
>insert into emppartition partition(mno) values(2,'jay','89272861');
>insert into emppartition partition(mno) values(3,'nikhil','47252258');

select * from emppartition;

2. Create a table emp_buck_no_partition with only bucketing on the m_no column and
load data into table.
$hive;
>Create bucketed table
create table emp_buck_no_part(
empid int,
name string,
mno string
) clustered by (mno) into 5 buckets;

>insert into emp_buck_no_part values(1,'abc','9654532165');
>insert into emp_buck_no_part values(2,'def','9589548213');
>insert into emp_buck_no_part values(3,'hgj','9589548213');
>insert into emp_buck_no_part values(4,'gog','9654532165');
>describe formatted emp_buck_no_part;
>select * from emp_buck_no_part;

3. Create emp_buck_with_part on mno column and load data into it.
$hive;
>create table emp_buck_part(
empid int,
name string,
mno string
) partitioned by (dept string) clustered by (mno) into 5 buckets row format delimited
fields terminated by ',';

>insert into emp_buck_part partition(dept='HR')values(1,'Alice','50000');

>insert into emp_buck_part partition(dept='IT') values(2,'Bob','60000');
>select * from emp_buck_part;

4.Create table empdata with partition(dept) and clustered by empid and load data into it.
>create table empdata(
empid int,
name string,
salary int
)
partitioned by (dept string)
clustered by (salary) into 4 buckets;

>insert into empdata partition(dept='HR') values(1,'Alice',50000);
>insert into empdata partition(dept='IT')values(2,'Bob',60000);
>select * from empdata;

5.Create table sales_data partitioned by year and month and load data into it.

>Create table
CREATE TABLE sales_data(
orderid INT,
product STRING,
amount DOUBLE
)
PARTITIONED BY (year INT, month INT);

>CREATE TABLE temp_sales(
orderid STRING,
product STRING,
amount DOUBLE,
month INT
);

>INSERT INTO temp_sales VALUES (1,'Laptop',1200,1);
>INSERT INTO temp_sales VALUES (2,'Keyboard',75,2);
>INSERT INTO temp_sales VALUES (3,'Monitor',300,2);
>INSERT INTO temp_sales VALUES (4,'Mouse',40,3);

>SET hive.exec.dynamic.partition.mode=nonstrict;
>Insert data into partition table
>INSERT INTO sales_data PARTITION(year=2023, month)
>SELECT orderid, product, amount, month
>FROM temp_sales;

>SELECT * FROM sales_data;
>SHOW PARTITIONS sales_data;

Practical No. 6: Demonstrating Publisher/Subscriber messaging system using 
Kafka. 
Step 1: Install Java jdk and make environment variable named JAVA_HOME with 
jdk path. 
Step 2: Paste “%JAVA_HOME\bin” in the system variables PATH.
Step 4: Open zookeeper.properties and changes its dataDir to 
“C:\kafka\zookeeper-data”. 
Step 5: Open server.properties and change its log.dirs to C:\kafka\kafka-logs. 
Step 3: Install “kafka_2.12-3.7.0” and extract its files in C:\kafka.
Step 6: Open Windows Powershell as administrator and run this command to start 
zookeeper server. 
cd C:\kafka 
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
 
Step 7: Open another Windows Powershell as administrator and run this command 
to start kafka server. 
cd C:\kafka 
.\bin\windows\kafka-server-start.bat .\config\server.properties
 
  
Step 8: Open Command Prompt with “C:\kafka\kafka_2.12-3.7.0\bin\windows” 
and run this command to create topic ‘test’. 
kafka-topics.bat --create --bootstrap-server localhost:9092 -
replication-factor 1 --partitions 1 --topic test
 
Step 9: Open another Command Prompt with “C:\kafka\kafka_2.12
3.7.0\bin\windows” and run this command to create producer. 
kafka-console-producer.bat --broker-list localhost:9092 --topic test
 Step 10: Open another Command Prompt with “C:\kafka\kafka_2.12
3.7.0\bin\windows” and run this command to create consumer. 
kafka-console-consumer.bat --bootstrap-server localhost:9092 --t
