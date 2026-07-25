import importlib
from typing import List
from omegaconf import OmegaConf
from .base import BaseTool


def get_tool_class(cls_name: str):
    """动态加载工具类"""
    module_name, class_name = cls_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def initialize_tools_from_config(config_path: str) -> List[BaseTool]:
    """从配置文件初始化工具列表
    
    配置文件格式:
        tools:
          - class_name: module.path.ToolClass
            config:
              param1: value1
    """
    if not config_path:
        return []
    
    cfg = OmegaConf.load(config_path)
    tools = []
    
    for tool_cfg in cfg.get("tools", []):
        cls = get_tool_class(tool_cfg.class_name)
        config = OmegaConf.to_container(tool_cfg.get("config", {}), resolve=True)
        tools.append(cls(config))
    
    return tools
