import h5py
import os


def save_results(acc, std_acc, auc, std_auc, loss, h5name, times):
    """保存单轮的acc、auc和loss到h5文件"""
    # 创建results目录（如果不存在）
    result_path = "../results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 构建文件名
    file_name = f"{h5name}_{times}.h5"
    file_path = os.path.join(result_path, file_name)

    # 保存数据
    with h5py.File(file_path, 'w') as hf:
        # 为每个列表创建对应的数据集，直接存储列表（h5py会自动处理为数组）
        hf.create_dataset('test_acc', data=acc)  # 存储acc列表
        hf.create_dataset('std_test_acc', data=std_acc)  # 存储std_acc列表
        hf.create_dataset('test_auc', data=auc)  # 存储auc列表
        hf.create_dataset('std_test_auc', data=std_auc)  # 存储std_auc列表
        hf.create_dataset('train_loss', data=loss)  # 存储loss列表

    print(f"Round {times} results saved to {file_path}")

