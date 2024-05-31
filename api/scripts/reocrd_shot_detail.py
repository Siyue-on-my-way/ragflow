import csv
import mysql.connector

# 配置数据库连接参数
# config = {
#     'user': 'your_username',
#     'password': 'your_password',
#     'host': 'localhost',
#     'database': 'your_database',
#     'raise_on_warnings': True
# }
config = {
    'user': 'root',
    'password': 'infini_rag_flow',
    'host': '118.178.241.227',
    'port': 5455,
    'database': 'rag_flow',
    'raise_on_warnings': True
}

def mysql_conn():
    """连接mysql数据库"""
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    return cnx, cursor

def mysql_create(cnx, cursor, create_table_query):
    # # 创建一个新表
    # cnx = mysql_conn() 
    cursor.execute(create_table_query)
    # 提交事务
    cnx.commit()
    print(f"Create [#table @{create_table_query}] success")
    
def mysql_handle(cnx, cursor, insert_query, data):
# 连接到MySQL数据库
    try:
        # cnx = mysql_conn()
        # cursor = cnx.cursor()
        # # 创建一个新表
        # create_table_query = """
        # CREATE TABLE IF NOT EXISTS users (
        #     id INT AUTO_INCREMENT PRIMARY KEY,
        #     username VARCHAR(255) NOT NULL,
        #     email VARCHAR(255)
        # )
        # """
        # cursor.execute(create_table_query)

        # 插入数据到表中    
        cursor.execute(insert_query, data)

        # 提交事务
        cnx.commit()
        print("数据插入成功")
        import time
        time.sleep(0.3)

    except mysql.connector.Error as err:
        print(f"数据库操作出错：{err}")
            

def read_csv_data(cnx, cursor, file_path):
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='$')
            
            # 跳过标题行（如果有）
            next(csv_reader)
            
            # 循环遍历CSV文件中的每一行
            insert_query = """
                INSERT INTO document_extra (pubid, title, summary) VALUES (%s, %s, %s)
                """  
            for row in csv_reader:
                # row是一个列表，包含了当前行的所有数据
                # 例如，print(row)会打印整行数据
                # 访问特定列的数据，例如第一列：row[0]
                
                # 这里可以添加你的逻辑来处理每一行的数据
                print(row)  # 打印整行数据作为示例
                data = (row[0], row[1], row[2])
                mysql_handle(cnx, cursor, insert_query=insert_query, data=data)

if __name__ == "__main__":
    cnx, cursor = mysql_conn()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS document_extra (
        id INT AUTO_INCREMENT PRIMARY KEY,
        pubid VARCHAR(25) NOT NULL,
        title VARCHAR(255) NOT NULL,
        summary TEXT, 
        UNIQUE (pubid)
    )"""
    # mysql_create(cnx, cursor,  create_table_query)
    file_path = 'patent.csv'
    read_csv_data(cnx, cursor, file_path=file_path)
    
    if cnx.is_connected():
        cursor.close()
        cnx.close()
        print("数据库连接已关闭")