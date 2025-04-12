"""
数据加载器，用于加载训练和评估数据。
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger("brainforge.data.loader")


def load_data(
        data_path: str,
        format_hint: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    加载数据文件

    参数:
        data_path: 数据文件路径
        format_hint: 格式提示(json, csv, jsonl)

    返回:
        数据列表，每项为包含"input"和"output"的字典
    """
    logger.info(f"Loading data from {data_path}")

    # 检查文件是否存在
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 确定文件格式
    if format_hint is None:
        _, ext = os.path.splitext(data_path)
        format_hint = ext.lstrip('.').lower()

    # 根据格式加载数据
    if format_hint == 'json':
        return _load_json(data_path)
    elif format_hint == 'jsonl':
        return _load_jsonl(data_path)
    elif format_hint == 'csv':
        return _load_csv(data_path)
    else:
        logger.warning(f"Unknown format: {format_hint}, trying as JSON")
        return _load_json(data_path)


def _load_json(file_path: str) -> List[Dict[str, str]]:
    """加载JSON格式数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保数据是列表格式
        if not isinstance(data, list):
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            else:
                logger.error(f"Invalid JSON format in {file_path}")
                raise ValueError(f"Invalid JSON format in {file_path}")

        # 验证每个数据项
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.error(f"Item {i} is not a dictionary")
                raise ValueError(f"Item {i} is not a dictionary")

            if 'input' not in item or 'output' not in item:
                logger.error(f"Item {i} missing required 'input' or 'output' fields")
                raise ValueError(f"Item {i} missing required 'input' or 'output' fields")

        logger.info(f"Loaded {len(data)} items from JSON file")
        return data

    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        raise


def _load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """加载JSONL格式数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        logger.warning(f"Line {i + 1} is not a valid JSON object, skipping")
                        continue

                    if 'input' not in item or 'output' not in item:
                        logger.warning(f"Line {i + 1} missing required 'input' or 'output' fields, skipping")
                        continue

                    data.append(item)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {i + 1}, skipping")

        logger.info(f"Loaded {len(data)} items from JSONL file")
        return data

    except Exception as e:
        logger.error(f"Error loading JSONL file: {str(e)}")
        raise


def _load_csv(file_path: str) -> List[Dict[str, str]]:
    """加载CSV格式数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # 确认必要的列
            if 'input' not in reader.fieldnames or 'output' not in reader.fieldnames:
                logger.error(f"CSV file missing required columns: 'input' or 'output'")
                raise ValueError(f"CSV file missing required columns: 'input' or 'output'")

            for i, row in enumerate(reader):
                item = {
                    'input': row['input'],
                    'output': row['output']
                }

                # 可选地添加其他列
                for key, value in row.items():
                    if key not in ['input', 'output']:
                        item[key] = value

                data.append(item)

        logger.info(f"Loaded {len(data)} items from CSV file")
        return data

    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise