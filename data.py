import csv
from faker import Faker

fake = Faker()

fields = ['Vehicle_ID', 'Customer_Name', 'Phone_Number', 'Email', 'Address']

rows = []

for i in range(1, 501):
    vehicle_id = f"V{i:04d}"
    customer_name = fake.name()
    phone_number = fake.msisdn()[3:13]  # 10 digit number
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    rows.append([vehicle_id, customer_name, phone_number, email, address])

with open('customer_data_500.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(rows)

print("CSV file 'customer_data_500.csv' created with 500 customer records.")
