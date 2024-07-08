import os

import requests
import json
from common.config_parse import ConfigParse
from common.file_path import FilePath


class VearchClient:
    _config_parser = ConfigParse(FilePath.config_file_path)
    master_url = _config_parser.get_vearch_master_server()
    router_url = _config_parser.get_vearch_router_server()
    space_config = 'space1'

    def get_cluster_stats(self):
        url = f'{self.master_url}/_cluster/stats'
        response = requests.get(url)
        return response

    def get_cluster_health(self):
        url = f'{self.master_url}/_cluster/health'
        response = requests.get(url)
        return response

    def get_servers_status(self):
        url = f'{self.master_url}/list/server'
        response = requests.get(url)
        return response

    def list_dbs(self):
        url = f'{self.master_url}/list/db'
        response = requests.get(url)
        return response

    def create_db(self, db_name):
        url = f"{self.master_url}/db/_create"
        data = {
            "name": db_name
        }
        data_json = json.dumps(data)
        response = requests.put(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    def get_db(self, db_name):
        url = f'{self.master_url}/db/{db_name}'
        response = requests.get(url)
        return response

    def drop_db(self, db_name):
        url = f'{self.master_url}/db/{db_name}'
        response = requests.delete(url)
        return response

    def list_spaces(self, db_name):
        url = f'{self.master_url}/list/space?db={db_name}'
        response = requests.get(url)
        return response

    def create_space(self, db_name, space_name, dimension):
        url = f"{self.master_url}/space/{db_name}/_create"
        data = {
            "name": space_name,
            "partition_num": 1,
            "replica_num": 1,
            "engine": {
                "name": "gamma",
                "index_size": 1,
                "id_type": "String",
                "retrieval_type": "HNSW",
                "retrieval_param": {
                    "metric_type": "InnerProduct",
                    "nlinks": 32,
                    "efConstruction": 140,
                    "efSearch": 64
                }
            },
            "properties": {
                "feature": {
                    "dimension": dimension,
                    "type": "vector"
                },
                "doc": {
                    "type": "string"
                },
                "relevant_doc": {
                    "type": "string"
                },
                "source": {
                    "type": "string"
                }
            }
        }
        data_json = json.dumps(data)
        response = requests.put(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    def create_space_new(self, db_name, space_name, dimension):
        url = f"{self.master_url}/space/{db_name}/_create"
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建config文件的路径
        config_path = os.path.join(current_dir, '..', '..', 'static', 'config', 'space_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        data = config[self.space_config]
        data['name'] = space_name
        data['properties']['feature']['dimension'] = dimension
        data_json = json.dumps(data)
        response = requests.put(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    def get_space(self, db_name, space_name):
        url = f"{self.master_url}/space/{db_name}/{space_name}"
        response = requests.get(url)
        return response

    def get_space_health(self, db_name, space_name):
        url = f"{self.master_url}/_cluster/health?db={db_name}&space={space_name}"
        response = requests.get(url)
        return response

    def drop_space(self, db_name, space_name):
        url = f"{self.master_url}/space/{db_name}/{space_name}"
        response = requests.delete(url)
        return response

    def insert_one(self, db_name, space_name, vec, doc, relevant_doc, source):
        url = f"{self.router_url}/{db_name}/{space_name}"
        data = {
            "feature": {
                "feature": vec
            },
            "doc": doc,
            "relevant_doc": relevant_doc,
            "source": source,
        }
        data_json = json.dumps(data)
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    def insert_one_new(self, db_name, space_name, vec, sentence, content, metadata):
        url = f"{self.router_url}/{db_name}/{space_name}"
        data = {
            'feature': {
                'feature': vec
            },
            'sentence': sentence,
            'content': content,
            'metadata': json.dumps(metadata, ensure_ascii=False),
        }
        data_json = json.dumps(data)
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    def insert_batch(self, db_name, space_name, data_list):
        """
        向指定的数据库和空间中批量插入向量数据表，建议批量插入不超过100条
        Args:
            db_name (str): 数据库名称
            space_name (str): 空间名称
            data_list (list): 包含要插入的数据的列表，格式如下
            [
                {
                    "_id": "1000000",
                    "field_int": 90399,
                    "field_float": 90399,
                    "field_double": 90399,
                    "field_string": "111399",
                    "field_vector": {
                        "feature": [...]  # 特征向量列表
                    }
                },
                {
                    "_id": "1000001",
                    "field_int": 45085,
                    "field_float": 45085,
                    "field_double": 45085,
                    "field_string": "106085",
                    "field_vector": {
                        "feature": [...]  # 特征向量列表
                    }
                }
            ]
        Returns:
            None
        """
        url = f'{self.router_url}/document/upsert'
        data = {
            "db_name": db_name,
            "space_name": space_name,
            "documents": data_list
        }
        for item in data["documents"]:
            item['metadata'] = json.dumps(item['metadata'], ensure_ascii=False)
        data_json = json.dumps(data)
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)
        return response

    # 根据向量相似度进行语义搜索
    def search_by_vec(self, db_name, space_name, vec, size=1):
        url = f"{self.router_url}/{db_name}/{space_name}/_search"
        data = {
            "query": {
                "sum": [{
                    "field": "feature",
                    "feature": vec,
                }]
            },
            "is_brute_search": 1,
            "retrieval_param": {
                "nprobe": 100
            },
            "size": size
        }

        data_json = json.dumps(data)
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)
        print(response.text)
        return response.json()

    def get_by_id(self, db_name, space_name, id):
        url = f"{self.router_url}/{db_name}/{space_name}/{id}"
        response = requests.get(url)
        return response
    
    def mget_by_ids(self, db_name, space_name, query):
        url = f'{self.router_url}/{db_name}/{space_name}/_query_byids'
        response = requests.post(url, json=query)
        return response
    
    def bulk_search(self, db_name, space_name, queries):
        url = f'{self.router_url}/{db_name}/{space_name}/_bulk_search'
        response = requests.post(url, json=queries)
        return response

    def msearch(self, db_name, space_name, query):
        url = f'{self.router_url}/{db_name}/{space_name}/_msearch'
        response = requests.post(url, json=query)
        return response

    def search_by_id_feature(self, db_name, space_name, query):
        url = f'{self.router_url}/{db_name}/{space_name}/_query_byids_feature'
        response = requests.post(url, json=query)
        return response

    def delete_by_query(self, db_name, space_name, query):
        url = f'{self.router_url}/{db_name}/{space_name}/_delete_by_query'
        response = requests.post(url, json=query)
        return response

    def delete_by_id(self, db_name, space_name, id):
        url = f'{self.router_url}/{db_name}/{space_name}/{id}'
        response = requests.delete(url)
        return response
