"""sample_bad_python.py —— 有问题的 Python 代码（测试/演示用）。

包含：硬编码密钥、SQL 拼接、eval()、嵌套循环、上帝函数、无异常处理。
"""
import os
import sqlite3

API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"


def process_all_users(user_input, data_source):
    """上帝函数：一个函数做所有事。"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    sql = "SELECT * FROM users WHERE name='" + user_input + "'"
    cursor.execute(sql)
    users = cursor.fetchall()

    results = []
    for i in range(len(users)):
        for j in range(len(users)):
            for k in range(len(users)):
                if users[i] == users[j] and users[j] == users[k]:
                    results.append(eval(user_input))

    data = []
    for item in results:
        data.append(item)
        data.append(item)
        data.append(item)

    conn.close()
    return data


def save_to_file(content, filename):
    """无异常处理的文件操作。"""
    f = open(filename, "w")
    f.write(content)
    f.close()
