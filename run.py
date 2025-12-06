from tools.decode_instructions import *

if __name__ == '__main__':
    # Step0: 收集处理数据集，搭建自己想要的模型，自己定义项目文件路径，使用模型等参数，以及根据需求构建流程

    # Step1: 创建新项目文件夹，同时获取参数和流程
    Config, Process, Step_name_list, Pipeline, Cwd = create_project_pipeline()

    # Step2: 复制基础执行文件和修改过后的参数文件，为修改训练检测代码做准备（根据数据集和模型修改）
    copy_train_test(Config, Cwd)

    # Step3: 按流程构建任务，并执行
    # 定义步骤依赖关系
    step_dependencies = {
        1: [],  # 步骤1（分割原始数据）没有依赖
        2: [],  # 步骤2（计算标准化参数）没有依赖
        3: [2],  # 步骤3（训练）依赖步骤2
        4: [3],  # 步骤4（检测）依赖步骤3
        5: [4]  # 步骤5（推理）依赖步骤4
    }
    # 执行流程并打印log
    subprocess_cmd_log(step_dependencies, Step_name_list, Pipeline, Cwd)
