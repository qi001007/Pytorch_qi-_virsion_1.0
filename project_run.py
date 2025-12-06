import os
import yaml
import subprocess
from shutil import copy
from pathlib import Path
from jinja2 import Template


# 创建新项目，同时获取参数和流程
def create_project_pipeline():
    # 构造当前工作目录Path
    cwd = Path.cwd()
    # 构造参数yaml的Path
    config_file = cwd / 'config.yaml'
    # 构造流程yaml的Path
    process_file = cwd / 'process.yaml'

    # 传入配置文件
    # 打开config和process为阅读模式，将yaml转化为py中的字典
    with open(config_file, 'r', encoding='utf-8') as c, open(process_file, 'r', encoding='utf-8') as p:
        config = yaml.safe_load(c)
        process = yaml.safe_load(p)

    # pipeline[step_name_list[i]][]来简化参数
    # 读取预设流程的步骤list
    step_name_list = [step['name'] for step in process['process']]
    # 构造新字典，将每一步step_name和内部参数对应方便调运
    pipeline = {step['name']: step for step in process['process']}

    # 创建并校验项目文件夹
    # 从config字典中获取提前定义的project_path
    project_path = Path(process['project_base']['project_path'])
    if not project_path.exists():
        try:
            # 创建项目目录
            project_path.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录
            print(f"项目文件夹已创建：{project_path}")
        except Exception as e:
            print(f"项目文件夹创建失败：{e}")
    else:
        print(f"项目文件夹已存在：{project_path}")

    # 改换工作目录(无论我的项目在哪里都没关系了)
    os.chdir(project_path)
    # 再次构造当前工作目录Path(更新)
    cwd = Path.cwd()
    print(f'当前项目文件夹: {cwd}')

    return config, process, step_name_list, pipeline, cwd


Config, Process, Step_name_list, Pipeline, Cwd = create_project_pipeline()


# 复制基础执行文件和修改过后的参数文件，为修改做准备
def copy_train_test(config):
    # 从传入的config中读取base_train_test中的train.py和test.py的名字和路径（还有base_config和base_process中的名字和路径），并将其重新写出字典方便调用
    name_path = {
        config['base_train_test']['base_train_name']:
            Path(config['base_train_test']['base_train_path']),
        config['base_train_test']['base_test_name']:
            Path(config['base_train_test']['base_test_path']),
        config['base_config']['base_config_name']:
            Path(config['base_config']['base_config_path']),
        config['base_process']['base_process_name']:
            Path(config['base_process']['base_process_path']),
    }
    # 循环从name_path中获取train.py和test.py的名字和路径来复制
    # .items() 是 Python 字典（dict） 的一个内置方法，用来一次性拿到“键 + 值”
    for name, path in name_path.items():
        if not (Cwd / name).exists():
            # 空项目中加入train.py和test.py（还有config.yaml,process.yaml）
            copy(path, Cwd / name)
            print(f'{name}创建成功')
        else:
            # 自己决定要不要覆盖
            mind = str(input(f'{name}已存在, 是否要覆盖 [y/n]')).lower()
            if mind == 'y':
                copy(path, Cwd / name)
                print(f'{name}覆盖成功')
            if mind == 'n':
                print(f'跳过覆盖{name}')
    # 每一个项目的train.py和test.py可能需要源码上的调整
    print(f"请根据需求改进{config['base_train_test']['base_train_name']}和{config['base_train_test']['base_test_name']}")


copy_train_test(Config)

def subprocess_cmd(step_name_list, pipeline, i):
    step_script = ''
    step_args = {}

    if 1 <= i <= len(step_name_list):
        # 获取这一步的运行指令
        step_script = pipeline[step_name_list[i-1]]['script']
        # 获取这一步的传入参数
        step_args = pipeline[step_name_list[i-1]]['args']
    else:
        print('参数i输入超限')
    # 根据运行指令和传入参数，构造字符串cmd指令
    cmd = step_script
    for parameters in step_args:
        # 获取原始占位串,将复杂的字典的[]变成了.,同时实现自己读自己
        template_str = step_args[parameters]
        # 在新字典中根据索引获取地址，避免重复更改
        real_path = Template(template_str).render(pipeline)
        cmd += f"  --{parameters} {Cwd / real_path}"
    print(cmd)

    # 执行指令
    subprocess.run(cmd, shell=True)


# subprocess_cmd(Step_name_list, Pipeline, i=1)
# subprocess_cmd(Step_name_list, Pipeline, i=2)
subprocess_cmd(Step_name_list, Pipeline, i=3)
# subprocess_cmd(Step_name_list, Pipeline, i=4)


