import sqlite3

DB_NAME = "users.db"

with sqlite3.connect(DB_NAME) as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM inventory")
    cursor.execute("DELETE FROM transactions")
    conn.commit()

print("All tables cleared successfully.")
