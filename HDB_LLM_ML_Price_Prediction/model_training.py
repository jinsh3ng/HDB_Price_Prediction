import pandas as pd
import psycopg2

conn = psycopg2.connect(database = "HDB", 
                        user = "postgres", 
                        host= 'localhost',
                        password = "admin",
                        port = 5432)

cur = conn.cursor()
cur.execute('SELECT * FROM resale_transactions;')

df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

print(df.head(5))
