import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "business.db"

random.seed(42)

REGIONS = [
    "North",
    "South",
    "East",
    "West",
]

SEGMENTS = [
    "Consumer",
    "SMB",
    "Enterprise",
]

CATEGORIES = [
    "Electronics",
    "Furniture",
    "Accessories",
]

PRODUCTS = [
    ("Laptop", "Electronics", 1200.00),
    ("Monitor", "Electronics", 350.00),
    ("Keyboard", "Accessories", 80.00),
    ("Mouse", "Accessories", 40.00),
    ("Desk", "Furniture", 450.00),
    ("Office Chair", "Furniture", 300.00),
    ("Headphones", "Accessories", 150.00),
    ("Tablet", "Electronics", 600.00),
]

def random_date(start, end):
    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))

def create_database():

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            region TEXT NOT NULL,
            segment TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id)
                REFERENCES customers(customer_id)
        );

        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id)
                REFERENCES orders(order_id),
            FOREIGN KEY (product_id)
                REFERENCES products(product_id)
        );
        """
    )

    customers = []

    for customer_id in range(1, 501):

        customers.append(
            (
                customer_id,
                f"Customer {customer_id}",
                random.choice(REGIONS),
                random.choice(SEGMENTS),
            )
        )

    cursor.executemany(
        """
        INSERT INTO customers (
            customer_id,
            customer_name,
            region,
            segment
        )
        VALUES (?, ?, ?, ?)
        """,
        customers,
    )

    products = []

    for product_id, (name, category, price) in enumerate(
        PRODUCTS,
        start=1,
    ):
        products.append(
            (
                product_id,
                name,
                category,
                price,
            )
        )

    cursor.executemany(
        """
        INSERT INTO products (
            product_id,
            product_name,
            category,
            price
        )
        VALUES (?, ?, ?, ?)
        """,
        products,
    )

    start_date = date(2024, 1, 1)
    end_date = date(2025, 12, 31)

    orders = []

    for order_id in range(1, 5001):

        customer_id = random.randint(
            1,
            500,
        )

        order_date = random_date(
            start_date,
            end_date,
        )

        orders.append(
            (
                order_id,
                customer_id,
                order_date.isoformat(),
            )
        )

    cursor.executemany(
        """
        INSERT INTO orders (
            order_id,
            customer_id,
            order_date
        )
        VALUES (?, ?, ?)
        """,
        orders,
    )

    order_items = []

    order_item_id = 1

    for order_id in range(1, 5001):

        number_of_products = random.randint(1, 4)

        selected_products = random.sample(
            range(1, len(PRODUCTS) + 1),
            number_of_products,
        )

        for product_id in selected_products:

            price = PRODUCTS[
                product_id - 1
            ][2]

            quantity = random.randint(1, 5)

            order_items.append(
                (
                    order_item_id,
                    order_id,
                    product_id,
                    quantity,
                    price,
                )
            )

            order_item_id += 1

    cursor.executemany(
        """
        INSERT INTO order_items (
            order_item_id,
            order_id,
            product_id,
            quantity,
            unit_price
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        order_items,
    )

    connection.commit()
    connection.close()

    print(f"Database created: {DB_PATH}")
    print(f"Customers: {len(customers)}")
    print(f"Products: {len(products)}")
    print(f"Orders: {len(orders)}")
    print(f"Order items: {len(order_items)}")

if __name__ == "__main__":
    create_database()
