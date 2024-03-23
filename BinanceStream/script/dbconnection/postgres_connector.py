import psycopg2

def postgres_connector():
    """
    Connect to the PostgreSQL database server
    :return: Connection object or None
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port="5432",
            database="Binance",
            user="postgres",
            password="@!AmauriPostG"
        )
        print("Connection to PostgreSQL DB successful")
    except Exception as e:
        print(f"The error '{e}' occurred")

    return conn