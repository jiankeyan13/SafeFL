import re
import wandb

# --- 配置区 ---
LOG_FILE = "0.25attack_2epoch.txt"  # 你的日志文件名
PROJECT_NAME = "FL-Test" # WandB 项目名
RUN_NAME = "cifar_0.25attack_2epoch"  # WandB 运行名

def parse_and_upload():
    # 1. 初始化 wandb
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    # 2. 读取文件内容
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        full_content = f.read()

    # 3. 按 "--- Round X ---" 切分日志，确保数据不会错位
    # 切分后，每个片段都属于一个特定的 Round
    segments = re.split(r'--- Round (\d+) ---', full_content)
    
    # segments 的结构大概是：[前缀, "0", "Round 0的内容", "1", "Round 1的内容", ...]
    # 我们从索引 1 开始，步长为 2 遍历
    for i in range(1, len(segments), 2):
        round_num = int(segments[i])
        block_content = segments[i+1]

        # 在这个 Round 的块里寻找 acc 和 asr
        acc_match = re.search(r'Global Eval \(Clean\): acc = ([\d.]+)', block_content)
        asr_match = re.search(r'Global Eval \(badnets_group\): asr=([\d.]+)', block_content)

        if acc_match and asr_match:
            acc = float(acc_match.group(1))
            asr = float(asr_match.group(1))

            # 4. 上传到 wandb
            wandb.log({
                "round": round_num,
                "clean_acc": acc,
                "badnets_asr": asr
            }, step=round_num)
            
            print(f"Uploaded Round {round_num}: acc={acc}, asr={asr}")
        else:
            print(f"Warning: Round {round_num} data incomplete, skipped.")

    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    parse_and_upload()