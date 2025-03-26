from neo4j import GraphDatabase
import py2neo

#与 Neo4j 的连接
uri = "bolt://localhost:7687"
username = "neo4j"
password = "lihong"

driver = GraphDatabase.driver(uri, auth=(username, password))

def run_cypher_query(query):
    with driver.session() as session:
        result = session.run(query)

        # 打印查询结果
        for record in result:
            print(record)

        # 将结果转换为字典列表
        records_as_dicts = [record.value for record in result]
        return records_as_dicts

def execute_cypher_from_file(file_path):
    with open(file_path, 'r') as file:
        cypher_queries = file.readlines()

    with driver.session() as session:
        for query in cypher_queries:
            query = query.strip()  # 去除行尾的换行符
            if not query:  # 忽略空行
                continue
            result = session.run(query)
            for record in result:
                print(record)  # 打印查询结果

file_path = 'cypher_inchikey_smile.txt'
execute_cypher_from_file(file_path)