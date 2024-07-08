import traceback
import requests
import os
import json
import logging
from flask import Flask, render_template, request, send_file, jsonify, Response
from src.generator.gen_case import gen_case_by_req, gen_case_by_recall
from src.embedding.vearch_client import VearchClient
from src.knowledge_injection.injection import inject_knowledge

app = Flask(__name__)
# 设置日志级别为 INFO
app.logger.setLevel(logging.INFO)

# 内存缓存
CACHE_FILE = os.path.expanduser('~/call_count_cache.json')
print(CACHE_FILE)
# 初始化缓存文件
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'w') as f:
        json.dump({}, f)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/workbench')
def workbench():
    return render_template('workbench.html')


@app.route('/genCasePrd', methods=['POST'])
def genCasePrd():
    """根据需求文档生成测试用例"""
    try:
        prompt = request.form['genCasePrd_Prompt']
        file = request.files['genCasePrd_File']
        case_type = request.form['genCasePrd_Dropdown']
        multimodal = request.form['genCasePrd_Dropdown_2']

        if case_type == 'excel':
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            extension = '.xlsx'
        elif case_type == 'markdown':
            mimetype = 'text/markdown'
            extension = '.md'
        else:  # Xmind
            mimetype = 'application/octet-stream'
            extension = '.xmind'
        # 记录方法调用次数
        increment_call_count(file.filename[:-4])
        return send_file(
            gen_case_by_req(
                fp=file,
                case_type=case_type,
                human_input=prompt,
                multimodal=multimodal
            ),
            as_attachment=True,
            download_name="{}{}".format(file.filename[:-4], extension),
            mimetype=mimetype
        )

    except ValueError as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def increment_call_count(file_name):
    with open(CACHE_FILE, 'r+') as f:
        cache = json.load(f)
        if file_name in cache:
            cache[file_name] += 1
        else:
            cache[file_name] = 1
        f.seek(0)
        json.dump(cache, f)
        f.truncate()


@app.route('/get_call_counts', methods=['GET'])
def get_call_counts():
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        # return jsonify(cache)
        total_sum = sum(cache.values())
        return jsonify({'total_sum': total_sum})
    except FileNotFoundError:
        app.logger.error(f"Can't not open cache file: {CACHE_FILE}")
        app.logger.error(traceback.format_exc())
        return None
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return None


@app.route('/genCaseTdd', methods=['POST'])
def genCaseTdd():
    """根据需求文档+召回文档生成测试用例"""
    space_name = request.form['genCaseTdd_SpaceName']
    prompt = request.form['genCaseTdd_Prompt']
    file = request.files['genCaseTdd_File']
    case_type = request.form['genCaseTdd_Dropdown']

    if case_type == 'excel':
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        extension = '.xlsx'
    elif case_type == 'markdown':
        mimetype = 'text/markdown'
        extension = '.md'
    else:  # Xmind
        mimetype = 'application/octet-stream'
        extension = '.xmind'

    try:
        return send_file(
            gen_case_by_recall(file=file, space_name=space_name, handwritten_text=prompt, case_type=case_type),
            as_attachment=True,
            download_name="{}{}".format(file.filename[:-4], extension),
            mimetype=mimetype
        )

    except ValueError as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/injectKnowledge', methods=['POST'])
def injectKnowledge():
    """知识注入"""
    space_name = request.form['Contact-1-Name']
    file = request.files['injectKnowledge_File']
    analyze_images = 'Contact-1-Checkbox' in request.form  # 若复选框未选中，则默认为 False

    try:
        vearch = VearchClient()
        response = vearch.get_space(db_name='llm_test_db_1', space_name=space_name)
        if response.json()['code'] == 200:
            response = inject_knowledge(file, space_name, analyze_images)
            return response.json()
        else:
            return response.json()

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/querySpace', methods=['GET'])
def querySpace():
    """查询向量表空间"""
    space_name = request.args.get('spaceName')

    try:
        vearch = VearchClient()
        if space_name == '':
            data = vearch.list_spaces(db_name='llm_test_db_1')
        else:
            data = vearch.get_space(db_name='llm_test_db_1', space_name=space_name)
        return data.json()

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/createSpace', methods=['POST'])
def createSpace():
    """创建向量表空间"""
    data = request.get_json()

    try:
        vearch = VearchClient()
        if data and 'spaceName' in data:
            space_name = data['spaceName']
            space_info = vearch.create_space_new(db_name='llm_test_db_1', space_name=space_name, dimension=768)
            return space_info.json()
        else:
            return jsonify({'code': 400, 'msg': 'No spaceName provided'}), 400

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/deleteSpace', methods=['POST'])
def deleteSpace():
    """删除向量表空间"""
    space_name = request.get_json()['spaceName']

    try:
        vearch = VearchClient()
        space_info = vearch.drop_space(db_name='llm_test_db_1', space_name=space_name)
        return space_info.json()

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
