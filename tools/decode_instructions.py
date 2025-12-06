import os
import sys
import yaml
import time
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


# Config, Process, Step_name_list, Pipeline, Cwd = create_project_pipeline()


# 复制基础执行文件和修改过后的参数文件，为修改做准备
def copy_train_test(config, cwd):
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
        if not (cwd / name).exists():
            # 空项目中加入train.py和test.py（还有config.yaml,process.yaml）
            copy(path, cwd / name)
            print(f'{name}创建成功')
        else:
            # 自己决定要不要覆盖
            mind = str(input(f'{name}已存在, 是否要覆盖 [y/n]')).lower()
            if mind == 'y':
                copy(path, cwd / name)
                print(f'{name}覆盖成功')
            if mind == 'n':
                print(f'跳过覆盖{name}')
    # 每一个项目的train.py和test.py可能需要源码上的调整
    print(f"请根据需求改进{config['base_train_test']['base_train_name']}和{config['base_train_test']['base_test_name']}")


# copy_train_test(Config, Cwd)


def subprocess_cmd(cwd, step_name_list, pipeline, i):
    step_script = ''
    step_args = {}

    if 1 <= i <= len(step_name_list):
        # 获取这一步的运行指令
        step_script = pipeline[step_name_list[i - 1]]['script']
        # 获取这一步的传入参数
        step_args = pipeline[step_name_list[i - 1]]['args']
        # 获取步骤名称用于日志
        step_name = step_name_list[i - 1]
    else:
        print('参数i输入超限')
        return False
    # 根据运行指令和传入参数，构造字符串cmd指令
    cmd = 'python ' + str(cwd / step_script)
    # cmd = 'python ' + step_script
    for parameters in step_args:
        # 获取原始占位串,将复杂的字典的[]变成了.,同时实现自己读自己
        template_str = step_args[parameters]
        # 在新字典中根据索引获取地址，避免重复更改
        real_path = Template(template_str).render(pipeline)
        cmd += f"  --{parameters} {cwd / real_path}"
        # cmd += f"  --{parameters} {real_path}"
    print(f"执行命令: {cmd}\n")

    # 执行指令
    try:
        start_time = time.time()
        print(f"--- 开始执行步骤 {i}: {step_name} ---")

        # 方法1: 直接传递输出到终端（保持进度条效果）
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=None,  # 直接输出到终端
            stderr=None,  # 直接输出到终端
            text=True,
            encoding='utf-8'
        )

        execution_time = time.time() - start_time
        print(f"\n步骤完成时间: {execution_time:.2f}秒")

        # 检查执行结果
        if result.returncode == 0:
            print(f"✓ 步骤 {i} ({step_name}) 执行成功\n")
            return True
        else:
            print(f"✗ 步骤 {i} ({step_name}) 执行失败!")
            print(f"错误码: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ 步骤 {i} ({step_name}) 执行超时!")
        return False
    except Exception as e:
        print(f"✗ 步骤 {i} ({step_name}) 执行异常: {str(e)}")
        return False


def check_step_completion(cwd, step_name):
    """检查前一步是否完成"""
    flag_file = cwd / f".{step_name}_completed.flag"
    return flag_file.exists()


def mark_step_completed(cwd, step_name):
    """标记步骤完成"""
    flag_file = cwd / f".{step_name}_completed.flag"
    flag_file.touch()
    print(f"✓ 已标记 {step_name} 完成")


def cleanup_flag_files(cwd, step_name_list):
    """清理所有flag文件"""
    print("\n" + "="*50)
    print("清理flag文件...")
    removed_count = 0
    for step_name in step_name_list:
        flag_file = cwd / f".{step_name}_completed.flag"
        if flag_file.exists():
            try:
                flag_file.unlink()
                removed_count += 1
                print(f"已删除: {flag_file.name}")
            except Exception as e:
                print(f"删除 {flag_file.name} 失败: {e}")
    print(f"共清理了 {removed_count} 个flag文件")
    print("="*50 + "\n")


def subprocess_cmd_log(step_dependencies, step_name_list, pipeline, cwd):
    all_steps_successful = True

    try:
        # sorted() 是Python内置的一个函数，用于对可迭代对象（如列表、元组、字典等）进行排序，并返回一个新的已排序列表
        for step_num, dependencies in sorted(step_dependencies.items()):
            step_name = step_name_list[step_num - 1]

            # 检查依赖是否满足
            deps_met = all(
                check_step_completion(cwd, step_name_list[dep - 1])
                for dep in dependencies
            )

            if not deps_met and dependencies:
                print(f"步骤 {step_num} ({step_name}) 的依赖步骤未完成!")
                print(f"缺少的依赖: {dependencies}")
                all_steps_successful = False
                break

            # 执行步骤
            print(f"\n{'=' * 60}")
            print(f"开始执行步骤 {step_num}: {step_name}")
            print(f"{'=' * 60}")
            success = subprocess_cmd(cwd, step_name_list, pipeline, i=step_num)

            if success:
                mark_step_completed(cwd, step_name)
            else:
                print(f"步骤 {step_num} 失败，停止执行")
                all_steps_successful = False
                break

    finally:
        # 无论成功还是失败，都尝试清理flag文件
        if all_steps_successful:
            print("\n✓ 所有步骤执行完成！")
        else:
            print("\n✗ 流程执行失败！")

        # 清理flag文件
        cleanup_flag_files(cwd, step_name_list)

        if not all_steps_successful:
            sys.exit(1)

# print('当前工作路径：{}'.format(os.getcwd()))
# subprocess_cmd(Cwd, Step_name_list, Pipeline, i=1)
# subprocess_cmd(Cwd, Step_name_list, Pipeline, i=2)
# subprocess_cmd(Cwd, Step_name_list, Pipeline, i=3)
# subprocess_cmd(Cwd, Step_name_list, Pipeline, i=4)
# subprocess_cmd(Cwd, Step_name_list, Pipeline, i=5)
