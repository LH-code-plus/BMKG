import requests
from bs4 import BeautifulSoup
import pandas as pd

def query_uniprot_by_ec(ec_number):
    """
    根据 EC number 在 UniProt 数据库中查询 UniProtID。

    Args:
        ec_number: EC number.

    Returns:
        list: 包含 UniProtID 的列表。
    """

    url = f"https://enzyme.expasy.org/EC/{ec_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    uniprot_ids = []
    for result in soup.find_all('div', class_='entry'):
        uniprot_id_div = result.find('a', class_='primary-identifier')
        if uniprot_id_div:
            uniprot_id = uniprot_id_div.text
            uniprot_ids.append(uniprot_id)

    return uniprot_ids

def query_uniprot_by_id(uniprot_id):
    """
    根据 UniProtID 在 UniProt 数据库中查询序列。

    Args:
        uniprot_id: UniProtID.

    Returns:
        str: 蛋白质序列。
    """

    url = f"https://www.uniprot.org/uniprot/{uniprot_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    sequence_div = soup.find('div', class_='sequence')
    if sequence_div:
        return sequence_div.text.strip()
    else:
        return None

def main():
    # 读取 CSV 文件
    df = pd.read_csv('../datas/EC_number_Sequence.csv')
    ec_numbers = df['EC_number'].tolist()

    # 初始化结果列表
    results = []

    for ec_number in ec_numbers:
        uniprot_ids = query_uniprot_by_ec(ec_number)
        for uniprot_id in uniprot_ids:
            sequence = query_uniprot_by_id(uniprot_id)
            if sequence:
                results.append({'EC_number': ec_number, 'UniProtID': uniprot_id, 'sequence': sequence})
                print(f"For EC number {ec_number} and UniProtID {uniprot_id}, sequence found.")
            else:
                print(f"No sequence found for UniProtID {uniprot_id}")

    # 保存结果到 CSV 文件
    df = pd.DataFrame(results)
    df.to_csv('../datas/uniprot_results.csv', index=False)

if __name__ == "__main__":
    main()