# =====================================================================
# Garage Data System: PySpark Analysis
# =====================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from num2words import num2words # You may need to run: pip install num2words

# ---------------------------------------------------------------------
# 1. Spark Session and Data Loading
# ---------------------------------------------------------------------
spark = SparkSession.builder.appName("GarageAnalysis").getOrCreate()

# Load data into DataFrames
customer_df = spark.read.csv("customer_table.csv", header=True, inferSchema=True)
ser_det_df = spark.read.csv("ser_det_table.csv", header=True, inferSchema=True)
employee_df = spark.read.csv("employee_table.csv", header=True, inferSchema=True)
sparepart_df = spark.read.csv("sparepart_table.csv", header=True, inferSchema=True)
purchase_df = spark.read.csv("purchase_table.csv", header=True, inferSchema=True)
vendor_df = spark.read.csv("vendor_table.csv", header=True, inferSchema=True)

# ---------------------------------------------------------------------
# 2. Answering Questions
# ---------------------------------------------------------------------

# Q.1 List all the customers serviced.
print("--- Q.1 List all the customers serviced ---")
customer_df.join(ser_det_df, customer_df.cid == ser_det_df.cid, "inner") \
    .select("cname").distinct().show()

# Q.2 Customers who are not serviced.
print("--- Q.2 Customers who are not serviced ---")
customer_df.join(ser_det_df, customer_df.cid == ser_det_df.cid, "left_anti") \
    .select("cname").show()

# Q.3 Employees who have not received the commission.
print("--- Q.3 Employees who have not received the commission ---")
employee_df.join(ser_det_df, employee_df.eid == ser_det_df.eid) \
    .where(F.col("comm") == 0).select("ename").distinct().show()

# Q.4 Name the employee who have maximum Commission.
print("--- Q.4 Name the employee who have maximum Commission ---")
max_comm = ser_det_df.agg(F.max("comm")).collect()[0][0]
employee_df.join(ser_det_df, employee_df.eid == ser_det_df.eid) \
    .where(F.col("comm") == max_comm).select("ename").show()

# Q.5 Show employee name and minimum commission amount received by an employee.
print("--- Q.5 Show employee name and minimum commission amount received ---")
min_comm = ser_det_df.where(F.col("comm") > 0).agg(F.min("comm")).collect()[0][0]
employee_df.join(ser_det_df, employee_df.eid == ser_det_df.eid) \
    .where(F.col("comm") == min_comm).select("ename", "comm").show()

# Q.6 Display the Middle record from any table.
print("--- Q.6 Display the Middle record from ser_det_table ---")
total_count = ser_det_df.count()
middle_row_index = (total_count // 2) + 1
window_spec = Window.orderBy("sid")
ser_det_df.withColumn("rn", F.row_number().over(window_spec)) \
    .where(F.col("rn") == middle_row_index).drop("rn").show()

# Q.7 Display last 4 records of any table.
print("--- Q.7 Display last 4 records of employee_table ---")
employee_df.orderBy(F.col("eid").desc()).limit(4).show()

# Q.8 Count the number of records without count function from any table.
print("--- Q.8 Count the number of records without count function ---")
# This is tricky without count(). A common way is to add a row number and find the max.
window_spec = Window.orderBy("eid")
count_val = employee_df.withColumn("rn", F.row_number().over(window_spec)).agg(F.max("rn")).collect()[0][0]
print(f"Record count for employee_table: {count_val}")

# Q.9 Delete duplicate records from "Ser_det" table on cid.
print("--- Q.9 Show unique records from ser_det_table based on cid ---")
ser_det_df.dropDuplicates(["cid"]).show()

# Q.10 Show the name of Customer who have paid maximum amount.
print("--- Q.10 Customer who paid maximum amount ---")
max_total = ser_det_df.agg(F.max("total")).collect()[0][0]
customer_df.join(ser_det_df, customer_df.cid == ser_det_df.cid) \
    .where(F.col("total") == max_total).select("cname").show()

# Q.11 Display Employees who are not currently working.
print("--- Q.11 Employees who are not currently working ---")
employee_df.where(F.col("edol").isNotNull()).select("ename").show()

# Q.12 How many customers serviced their two wheelers.
print("--- Q.12 How many customers serviced their two wheelers ---")
ser_det_df.where(F.lower(F.col("type_veh")) == "two wheeler") \
    .select("cid").distinct().count()

# Q.13 List the Purchased Items which are used for Customer Service with Unit of that Item.
print("--- Q.13 Purchased Items used for Customer Service with Unit ---")
sparepart_df.join(ser_det_df, sparepart_df.spid == ser_det_df.spid, "inner") \
    .select("spname", "spunit").distinct().show()

# Q.14 Customers who have Colored their vehicles.
print("--- Q.14 Customers who have Colored their vehicles ---")
customer_df.join(ser_det_df, customer_df.cid == ser_det_df.cid) \
    .where(F.lower(F.col("typ_ser")) == "color").select("cname").distinct().show()

# Q.15 Find the annual income of each employee inclusive of Commission.
print("--- Q.15 Annual income of each employee ---")
commission_agg = ser_det_df.groupBy("eid").agg(F.sum("comm").alias("total_comm"))
employee_df.join(commission_agg, "eid", "left") \
    .na.fill(0, ["total_comm"]) \
    .withColumn("annual_income", (F.col("esal") * 12) + F.col("total_comm")) \
    .select("ename", "annual_income").show()

# Q.16 Vendor Names who provides the engine oil.
print("--- Q.16 Vendors who provide engine oil ---")
vendor_df.join(purchase_df, "vid").join(sparepart_df, "spid") \
    .where(F.col("spname").like("%ENGINE OIL%")).select("vname").distinct().show()

# Q.17 Total Cost to purchase the Color and name the color purchased.
print("--- Q.17 Total Cost to purchase Color ---")
purchase_df.join(sparepart_df, "spid") \
    .where(F.col("spname").like("%COLOUR%")) \
    .groupBy("spname").agg(F.sum("total").alias("total_purchase_cost")).show()

# Q.18 Purchased Items which are not used in "Ser_det".
print("--- Q.18 Purchased Items not used in Service ---")
purchased_spids = purchase_df.select("spid").distinct()
serviced_spids = ser_det_df.select("spid").distinct()
unused_spids = purchased_spids.subtract(serviced_spids)
unused_spids.join(sparepart_df, "spid").select("spname").show()

# Q.19 Spare Parts Not Purchased but existing in Sparepart.
print("--- Q.19 Spare Parts Not Purchased ---")
sparepart_df.join(purchase_df, "spid", "left_anti").select("spname").show()

# Q.20 Calculate the Profit/Loss of the Firm.
print("--- Q.20 Calculate Profit/Loss ---")
total_revenue = ser_det_df.agg(F.sum("total")).collect()[0][0]
total_salary_cost = employee_df.agg(F.sum("esal")).collect()[0][0]
total_purchase_cost = purchase_df.agg(F.sum("total")).collect()[0][0]
profit_loss = total_revenue - (total_salary_cost + total_purchase_cost)
print(f"Firm Profit/Loss: {profit_loss}")

# Q.21 Specify the names of customers who have serviced their vehicles more than one time.
print("--- Q.21 Customers with more than one service ---")
customer_counts = ser_det_df.groupBy("cid").count().where(F.col("count") > 1)
customer_counts.join(customer_df, "cid").select("cname").show()

# Q.22 List the Items purchased from vendors locationwise.
print("--- Q.22 Items purchased location-wise ---")
vendor_df.join(purchase_df, "vid").join(sparepart_df, "spid") \
    .select(F.col("vadd").alias("location"), F.col("spname").alias("item_name")) \
    .orderBy("location").show()

# Q.23 Display count of two wheeler and four wheeler from ser_details.
print("--- Q.23 Count of two and four wheelers serviced ---")
ser_det_df.groupBy("type_veh").count().show()

# Q.24 Display name of customers who paid highest SPGST and for which item.
print("--- Q.24 Customers serviced with highest SPGST item ---")
max_spgst = purchase_df.agg(F.max("spgst")).collect()[0][0]
spid_with_max_spgst = purchase_df.where(F.col("spgst") == max_spgst).select("spid")
ser_det_df.join(spid_with_max_spgst, "spid") \
    .join(customer_df, "cid").join(sparepart_df, "spid") \
    .select("cname", "spname").distinct().show()

# Q.25 Display vendors name who have charged highest SPGST rate for which item.
print("--- Q.25 Vendor with highest SPGST ---")
purchase_df.where(F.col("spgst") == max_spgst) \
    .join(vendor_df, "vid").join(sparepart_df, "spid") \
    .select("vname", "spname").show()
    
# Q.26 List name of item and employee name who have received item.
print("--- Q.26 Item and employee who used it in service ---")
employee_df.join(ser_det_df, "eid").join(sparepart_df, "spid") \
    .select("ename", "spname").distinct().show()

# Q.27 Display Customer Name, Vehicle Number, Item Used, Purchase Date, Vendor, and Location.
print("--- Q.27 Detailed service report ---")
ser_det_df.join(customer_df, "cid") \
    .join(sparepart_df, "spid") \
    .join(purchase_df, "spid") \
    .join(vendor_df, "vid") \
    .select("cname", "veh_no", "spname", "pdate", "vname", "vadd").distinct().show()

# Q.28 who belong this vehicle 'MH14PA335'?
print("--- Q.28 Owner of vehicle MH14PA335 ---")
ser_det_df.where(F.col("veh_no") == 'MH14PA335').join(customer_df, "cid") \
    .select("cname").show()

# Q.29 Display the name of customer from New York and their vehicle service date.
print("--- Q.29 Service date for New York customers ---")
customer_df.where(F.col("cadd") == 'NEW YORK').join(ser_det_df, "cid") \
    .select("cname", "ser_date").show()

# Q.30 from whom we have purchased items having maximum cost?
print("--- Q.30 Vendor for max cost purchase ---")
max_purchase_total = purchase_df.agg(F.max("total")).collect()[0][0]
purchase_df.where(F.col("total") == max_purchase_total).join(vendor_df, "vid") \
    .select("vname").show()

# Q.31 Display employees who are not 'Mechanic' and have done services.
print("--- Q.31 Non-mechanic employees who did services ---")
employee_df.where(F.col("ejob") != 'MECHANIC').join(ser_det_df, "eid", "inner") \
    .select("ename").distinct().show()

# Q.32 Display jobs with more than two employees.
print("--- Q.32 Jobs with more than two employees ---")
employee_df.groupBy("ejob").count().where(F.col("count") > 2).show() # > 1 for sample data

# Q.33 Display details of employees who did a service and rank them by number of services.
print("--- Q.33 Rank employees by service count ---")
service_counts = ser_det_df.groupBy("eid").count()
window_spec = Window.orderBy(F.col("count").desc())
employee_df.join(service_counts, "eid").select("ename", "ejob", "count") \
    .withColumn("service_rank", F.rank().over(window_spec)).show()

# Q.34 Display painter/fitter employees who provided a service and total service count for each.
print("--- Q.34 Service counts for Painters and Fitters ---")
employee_df.where(F.col("ejob").isin(['PAINTER', 'FITTER'])) \
    .join(ser_det_df, "eid", "inner").groupBy("ename", "ejob").count().show()

# Q.35 Display employee salary and provide a Grade based on salary.
print("--- Q.35 Grade employees based on salary ---")
window_spec = Window.orderBy(F.col("esal").desc().nulls_last())
employee_df.withColumn("rank", F.dense_rank().over(window_spec)) \
    .withColumn("salary_grade", F.when(F.col("rank") <= 2, 'A')
                                   .when(F.col("rank") <= 4, 'B')
                                   .otherwise('C')) \
    .select("ename", "esal", "salary_grade").show()

# Q.36 display the 4th record of emp table without using group by and rowid.
print("--- Q.36 Display the 4th record of employee table ---")
window_spec = Window.orderBy("eid")
employee_df.withColumn("rn", F.row_number().over(window_spec)) \
    .where(F.col("rn") == 4).drop("rn").show()

# Q.37 Provide a commission 100 to employees who are not earning any commission.
print("--- Q.37 Show proposed new commission ---")
ser_det_df.withColumn("new_commission", 
    F.when(F.col("comm") == 0, 100).otherwise(F.col("comm"))) \
    .select("sid", "eid", "comm", "new_commission").show()

# Q.38 totals no. of services for each day and place the results in descending order.
print("--- Q.38 Total services per day ---")
ser_det_df.groupBy("ser_date").count().orderBy(F.col("count").desc()).show()

# Q.39 Display the service details of those customer who belong from same city.
print("--- Q.39 Services for customers from cities with multiple customers ---")
city_counts = customer_df.groupBy("cadd").count().where(F.col("count") > 1)
multi_cust_cities = city_counts.select("cadd")
customer_df.join(multi_cust_cities, "cadd").join(ser_det_df, "cid") \
    .select(ser_det_df["*"]).show()

# Q.40 find all pairs of customers serviced by a single employee.
print("--- Q.40 Pairs of customers serviced by the same employee ---")
s1 = ser_det_df.alias("s1")
s2 = ser_det_df.alias("s2")
c1 = customer_df.alias("c1")
c2 = customer_df.alias("c2")
s1.join(s2, (s1.eid == s2.eid) & (s1.cid < s2.cid)) \
  .join(c1, s1.cid == c1.cid) \
  .join(c2, s2.cid == c2.cid) \
  .select(c1.cname.alias("customer1"), c2.cname.alias("customer2"), s1.eid) \
  .distinct().show()

# Q.41 List each service number follow by name of the customer who made that service.
print("--- Q.41 Service ID and Customer Name ---")
ser_det_df.join(customer_df, "cid").select("sid", "cname").show()

# Q.42 Provide employee rating (A,B,C,D) based on number of services.
print("--- Q.42 Employee rating based on service count ---")
service_counts_df = ser_det_df.groupBy("eid").count()
employee_df.join(service_counts_df, "eid", "left").na.fill(0) \
    .withColumn("rating", F.when(F.col("count") >= 3, 'A')
                            .when(F.col("count") == 2, 'B')
                            .when(F.col("count") == 1, 'C')
                            .otherwise('D')) \
    .select("ename", "ejob", "count", "rating").show()

# Q.43 Get maximum service amount of each customer with their customer details.
print("--- Q.43 Max service amount for each customer ---")
max_service_amt_df = ser_det_df.groupBy("cid").agg(F.max("total").alias("max_service_amount"))
customer_df.join(max_service_amt_df, "cid").show()

# Q.44 Get the details of customers with his total no of services.
print("--- Q.44 Total services for each customer ---")
total_services_df = ser_det_df.groupBy("cid").count().withColumnRenamed("count", "total_services")
customer_df.join(total_services_df, "cid", "left").na.fill(0).show()

# Q.45 From which location sparpart purchased with highest cost?
print("--- Q.45 Location of highest cost spare part purchase ---")
max_sprate = purchase_df.agg(F.max("sprate")).collect()[0][0]
purchase_df.where(F.col("sprate") == max_sprate).join(vendor_df, "vid").select("vadd").show()

# Q.46 Get the details of employee with their service details who has salary is null.
print("--- Q.46 Service details for employee with null salary ---")
employee_df.where(F.col("esal").isNull()).join(ser_det_df, "eid") \
    .select(employee_df.ename, employee_df.ejob, ser_det_df["*"]).show()

# Q.47 find the sum of purchase location wise.
print("--- Q.47 Sum of purchases by location ---")
vendor_df.join(purchase_df, "vid").groupBy("vadd") \
    .agg(F.sum("total").alias("total_purchase_amount")).show()

# Q.48 write a query sum of purchase amount in word location wise?
print("--- Q.48 Sum of purchases by location (in words) ---")
def number_to_words_py(n):
    if n is None:
        return None
    return num2words(int(n))

number_to_words_udf = F.udf(number_to_words_py, StringType())
purchase_sum_df = vendor_df.join(purchase_df, "vid").groupBy("vadd") \
    .agg(F.sum("total").alias("total_amount"))
purchase_sum_df.withColumn("amount_in_words", number_to_words_udf(F.col("total_amount"))).show()

# Q.49 Has the customer who has spent the largest amount money been given the highest rating.
print("--- Q.49 Check rating of top-spending customer ---")
spending_df = ser_det_df.groupBy("cid").agg(F.sum("total").alias("total_spent"))
window_spec = Window.orderBy(F.col("total_spent").desc())
rating_df = spending_df.withColumn("rank", F.dense_rank().over(window_spec)) \
    .withColumn("rating", F.when(F.col("rank") == 1, 'A')
                             .when(F.col("rank") <= 3, 'B')
                             .otherwise('C'))
customer_df.join(rating_df, "cid").orderBy(F.col("total_spent").desc()).show()

# Q.50 select the total amount in service for each customer for which the total is greater than the amount of the largest service amount in the table.
print("--- Q.50 Customer total spend > largest single service amount ---")
largest_service_amount = ser_det_df.agg(F.max("total")).first()[0]
customer_total_spend_df = ser_det_df.groupBy("cid").agg(F.sum("total").alias("customer_total_spend"))
customer_total_spend_df.where(F.col("customer_total_spend") > largest_service_amount) \
    .join(customer_df, "cid").select("cname", "customer_total_spend").show()

# Q.51 List the customer name and sparepart name used for their vehicle and vehicle type.
print("--- Q.51 Customer, spare part, and vehicle type ---")
ser_det_df.join(customer_df, "cid").join(sparepart_df, "spid") \
    .select("cname", "spname", "type_veh").show()

# Q.52 Get spname, ename, cname, quantity, rate, service amount for records in service table.
print("--- Q.52 Detailed service breakdown ---")
ser_det_df.join(sparepart_df, "spid").join(employee_df, "eid").join(customer_df, "cid") \
    .select("spname", "ename", "cname", "qty", "sp_rate", "ser_amt").show()

# Q.53 specify the vehicles owners who’s tube damaged.
print("--- Q.53 Owners of vehicles with tube damage ---")
ser_det_df.where(F.col("typ_ser") == 'TUBE DAMAGED').join(customer_df, "cid") \
    .select("cname").distinct().show()

# Q.54 Specify the details who have taken full service.
print("--- Q.54 Details of customers with full servicing ---")
ser_det_df.where(F.col("typ_ser") == 'FULL SERVICING').join(customer_df, "cid") \
    .select("cname", "cadd", "veh_no", "ser_date").show()

# Q.55 Select the employees who have not worked yet and left the job.
print("--- Q.55 Employees who left without doing any service ---")
serviced_employees = ser_det_df.select("eid").distinct()
employee_df.where(F.col("edol").isNotNull()) \
    .join(serviced_employees, "eid", "left_anti").select("ename").show()

# Q.56 Select employee who have worked first ever.
print("--- Q.56 Employee who performed the first-ever service ---")
first_service_date = ser_det_df.agg(F.min("ser_date")).first()[0]
ser_det_df.where(F.col("ser_date") == first_service_date).join(employee_df, "eid") \
    .select("ename").show()

# Q.57 Display all records falling in odd date.
print("--- Q.57 Service records on odd days of the month ---")
ser_det_df.where(F.dayofmonth(F.to_date(F.col("ser_date"), "dd-MMM-yy")) % 2 != 0).show()

# Q.58 Display all records falling in even date.
print("--- Q.58 Service records on even days of the month ---")
ser_det_df.where(F.dayofmonth(F.to_date(F.col("ser_date"), "dd-MMM-yy")) % 2 == 0).show()

# Q.59 Display the vendors whose material is not yet used.
print("--- Q.59 Vendors whose purchased material is unused ---")
used_spids = ser_det_df.select("spid").distinct()
purchased_but_unused_spids = purchase_df.join(used_spids, "spid", "left_anti").select("vid").distinct()
purchased_but_unused_spids.join(vendor_df, "vid").select("vname").show()

# Q.60 Difference between purchase date and used date of spare part.
print("--- Q.60 Time difference between part purchase and service use ---")
# Note: This will create duplicates if a part from multiple purchases is used.
ser_det_df.join(purchase_df, "spid").join(sparepart_df, "spid") \
    .withColumn("days_difference", F.datediff(
        F.to_date(F.col("ser_date"), "dd-MMM-yy"), 
        F.to_date(F.col("pdate"), "dd-MMM-yy"))) \
    .select("sid", "spname", "pdate", "ser_date", "days_difference").show()

# Stop the Spark Session
spark.stop()