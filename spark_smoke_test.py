# Step 1: Start SparkSession
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spark Smoke Test") \
    .config("spark.master", "local") \
    .getOrCreate()

print("SparkSession started successfully")

# Step 2: Create test DataFrame
data = [("Sandeep", 25), ("Alex", 30), ("Jordan", 22)]
columns = ["Name", "Age"]
df = spark.createDataFrame(data, columns)

print("DataFrame created:")
df.show()

# Step 3: Write to Parquet
df.write.mode("overwrite").parquet("test_output.parquet")
print("Parquet write successful")

# Step 4: Read back the Parquet file
df2 = spark.read.parquet("test_output.parquet")
print("Parquet read successful:")
df2.show()

# Step 5: Stop SparkSession
spark.stop()
print("SparkSession stopped")