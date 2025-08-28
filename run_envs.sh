#!/bin/bash

# --- Slurm 分布式配置 ---
#SBATCH -p amd_256
#SBATCH -N 1     
#SBATCH -n 64    

PYTHON_EXE="/public21/home/sc93921/.conda/envs/python/bin/python"
PORT=9997
NP=12             # 每个环境的MPI进程数
ENV_COUNT=5       # 环境数量
ENV_PREFIX="env"  # 环境文件夹前缀
RESTART_DELAY=30  # 环境失败后重启的延迟时间（秒）
PID_MAP_FILE=".pid_map" 
# 新增: 保留成功文件夹的数量
MAX_SUCCESS_FOLDERS=10

# --- 环境脚本路径 (重要) ---
ENV_SCRIPT="/public21/home/sc93921/software-sc93921/IBAMR/IBAMR-0.16.0/build3/env.sh"

# --- 颜色输出 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# --- Slurm 环境设置 ---
echo "--- Loading System and IBAMR Modules ---"
source ${ENV_SCRIPT}
echo "--- Verifying Correct Python Environment ---"
$PYTHON_EXE -c "import torch; print(f'PyTorch version found: {torch.__version__}')" || {
    echo -e "${RED}ERROR: Could not find torch using the specified Python. Exiting.${NC}"
    exit 1
}
echo "----------------------------------------"

# --- 清理旧成功文件夹函数 ---
cleanup_old_success_folders() {
    local env_id="$1"
    local env_dir="${env_id}"
    
    # 查找所有success_*文件夹并按数字排序
    local success_folders=($(find "$env_dir" -maxdepth 1 -name "success_*" -type d | sort -V))
    local folder_count=${#success_folders[@]}
    
    # 如果文件夹数量超过限制，删除最旧的
    if [ "$folder_count" -gt "$MAX_SUCCESS_FOLDERS" ]; then
        local folders_to_delete=$((folder_count - MAX_SUCCESS_FOLDERS))
        echo "  - Watchdog: Found $folder_count success folders, keeping latest $MAX_SUCCESS_FOLDERS"
        
        for ((i=0; i<folders_to_delete; i++)); do
            local folder_to_delete="${success_folders[$i]}"
            if [ -d "$folder_to_delete" ]; then
                if rm -rf "$folder_to_delete"; then
                    echo "  - Watchdog: 🗑️  Deleted old success folder: $(basename "$folder_to_delete")"
                else
                    echo "  - Watchdog: ⚠️  Failed to delete: $(basename "$folder_to_delete")"
                fi
            fi
        done
    fi
}

# --- Watchdog Function (适配分布式Slurm) ---
watchdog_task() {
    local log_file="$1"
    echo "👀 Watchdog task started in background, monitoring $log_file"
    
    local success_count=1
    touch "$log_file"

    tail -f -n 0 "$log_file" | stdbuf -oL grep --line-buffered -E "(回合失败:|回合成功:)" | while read -r line ; do
        
        local env_id=""
        if [[ $line =~ (回合失败|回合成功):[[:space:]]+([^,]+) ]]; then
            env_id="${BASH_REMATCH[2]}"
        fi

        if [ -z "$env_id" ]; then continue; fi

        # 从PID地图中查找要终止的ssh进程PID
        # 地图格式: "索引 PID 目录"
        local pid_to_kill=$(grep " ${env_id}$" "${PID_MAP_FILE}" | awk '{print $2}')

        if [ -z "$pid_to_kill" ]; then
            echo -e "  - ${YELLOW}Watchdog: Could not find PID for ${env_id} in ${PID_MAP_FILE}. It might have already been handled.${NC}"
            continue
        fi

        # 处理失败情况
        if [[ $line =~ 回合失败: ]]; then
            echo -e "${RED}🚨 Watchdog detected failure: $line${NC}"
            echo "  - Watchdog: Terminating ssh process ${pid_to_kill} for env ${env_id}."
            kill -9 "$pid_to_kill" 2>/dev/null
            echo "  - Watchdog: ✅ Termination signal sent to ssh PID ${pid_to_kill} (FAILURE)"
        
        # 处理成功情况
        elif [[ $line =~ 回合成功: ]]; then
            echo -e "${GREEN}🎉 Watchdog detected success: $line${NC}"
            
            # 重命名可视化文件夹
            local viz_dir="${env_id}/viz_eel2d_Str"
            if [ -d "$viz_dir" ]; then
                local new_name="${env_id}/success_${success_count}"
                if mv "$viz_dir" "$new_name"; then
                    echo "  - Watchdog: ✅ Renamed $viz_dir to $new_name"
                    success_count=$((success_count + 1))
                    
                    # 新增: 清理旧的成功文件夹，只保留最近10个
                    cleanup_old_success_folders "$env_id"
                fi
            fi
            
            # 终止进程以触发重启
            echo "  - Watchdog: Terminating ssh process ${pid_to_kill} for env ${env_id} to trigger restart."
            kill -9 "$pid_to_kill" 2>/dev/null
            echo "  - Watchdog: ✅ Termination signal sent to ssh PID ${pid_to_kill} (SUCCESS → RESTART)"
        fi
    done
}

# --- 清理函数 ---
final_cleanup() {
    echo -e "\n${YELLOW}=== Final Cleanup Triggered ===${NC}"
    # 终止所有由本脚本启动的后台作业
    # shellcheck disable=SC2046
    if ps -p $(jobs -p | tr '\n' ' ') > /dev/null; then kill $(jobs -p); fi
    scancel --job "${SLURM_JOBID}" # 确保所有slurm作业步被清理
    
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; fi
    if [ -n "$WATCHDOG_PID" ]; then kill "$WATCHDOG_PID" 2>/dev/null; fi
    rm -f "${PID_MAP_FILE}"
    echo "✓ All background tasks and jobs terminated."
}
trap final_cleanup EXIT TERM INT

# --- 脚本主逻辑 ---
PROJECT_ROOT=$(pwd) # 获取项目根目录的绝对路径
echo -e "${BLUE}=== Slurm Distributed Multi-Environment Runner (SSH-based) ===${NC}"
# 1. 节点分配
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${nodes[0]}
ALL_NODES=(${nodes[@]})
NUM_NODES=${#ALL_NODES[@]}

echo "Slurm Job ID: ${SLURM_JOB_ID}, Project Root: ${PROJECT_ROOT}"
echo -e "Head Node: ${GREEN}${HEAD_NODE}${NC}, All Nodes: ${GREEN}${ALL_NODES[*]}${NC}"

# 2. 获取头节点IP地址 (使用ssh)
echo -e "\n${YELLOW}Getting Head Node IP via ssh...${NC}"
HEAD_IP=$(ssh "$HEAD_NODE" "hostname -I" | awk '{print $1}')
if [ -z "$HEAD_IP" ]; then
    echo -e "${RED}Error: Could not determine Head Node IP.${NC}"
    exit 1
fi
export HEAD_NODE_IP=$HEAD_IP
echo -e "Head Node IP is: ${GREEN}${HEAD_NODE_IP}${NC}. Exported as environment variable."

# 3. 在头节点上启动Python服务器 (使用ssh)
echo -e "\n${YELLOW}Starting Python server on ${HEAD_NODE} via ssh...${NC}"
ssh "$HEAD_NODE" "
    source ${ENV_SCRIPT}
    cd ${PROJECT_ROOT}
    ${PYTHON_EXE} model_server.py --port ${PORT}
" > server.log 2>&1 &
SERVER_PID=$!
sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: Python server failed to start. Check server.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python server started (PID: ${SERVER_PID}) on ${HEAD_NODE}${NC}"

# 4. 启动Watchdog后台任务
echo -e "\n${YELLOW}Starting built-in Watchdog task...${NC}"
watchdog_task "model_server_log.txt" &
WATCHDOG_PID=$!
echo -e "${GREEN}✓ Watchdog task started in background (PID: ${WATCHDOG_PID})${NC}"

# 5. 初始化并并行启动所有环境 (使用ssh)
echo -e "\n${PURPLE}=== Launching all environments on all available nodes via ssh ===${NC}"
declare -a CMDS PIDS STATUS RESTART_TIME
# 清空PID地图文件
> "${PID_MAP_FILE}"

for i in $(seq 1 $ENV_COUNT); do
    env_dir="${ENV_PREFIX}${i}"
    # 以轮询方式为任务分配节点 (包括头节点)
    node_index=$(( (i-1) % NUM_NODES ))
    TARGET_NODE=${ALL_NODES[$node_index]}
    
    # 确定启动命令
    restart_subdir=$(find "$env_dir/restart_IB2dStrDiv" -name "restore.*" -type d 2>/dev/null | head -1)
    if [ -n "$restart_subdir" ]; then
        restart_full_name=$(basename "$restart_subdir")
        restart_name=${restart_full_name#restore.}
        CMDS[$i]="./main2d input2d restart_IB2dStrDiv $restart_name"
    else
        CMDS[$i]="./main2d input2d"
    fi
    echo -e "${BLUE}[$(date +%T)] Launching env ${i} on node ${TARGET_NODE}${NC}"
    
    # 使用ssh在目标节点后台启动mpiexec
    (
        log_file="env${i}.log"
        # 使用ssh在目标节点后台启动mpiexec, exec确保ssh进程取代子shell
        exec ssh "$TARGET_NODE" "
            source ${ENV_SCRIPT}
            export HEAD_NODE_IP=${HEAD_NODE_IP}
            cd '${PROJECT_ROOT}/${env_dir}'
            mpiexec --mca btl_openib_allow_ib true -np ${NP} ${CMDS[$i]}
        " >> "$log_file" 2>&1
    ) &
    
    PIDS[$i]=$!
    STATUS[$i]="running"
    # 记录 索引-PID-目录 映射
    echo "$i ${PIDS[$i]} ${env_dir}" >> "${PID_MAP_FILE}"
    echo -e "${GREEN}  ✓ Env ${i} launched (ssh PID: ${PIDS[$i]}) on ${TARGET_NODE}${NC}"

    # 新增：交错启动任务，给MPI运行时足够的初始化时间以避免冲突
    sleep 2
done

# 6. 实时监控与重启 (功能完全保留, 使用ssh重启)
echo -e "\n${PURPLE}=== Starting real-time monitoring & auto-restart (check every 60s) ===${NC}"
while true; do
    sleep 60
    
    active_envs_count=0
    for j in $(seq 1 $ENV_COUNT); do
        if [[ "${STATUS[$j]}" == "running" ]]; then
            active_envs_count=$((active_envs_count + 1))
        fi
    done
    
    if [ "$active_envs_count" -eq 0 ]; then
        echo -e "${GREEN}All environments appear to be finished or restarting. Exiting monitor loop.${NC}"
        break
    fi
    echo -e "\n${BLUE}[Monitor] $(date +%T) - Active environments: ${active_envs_count}/${ENV_COUNT}${NC}"
    
    for i in $(seq 1 $ENV_COUNT); do
        if [[ "${STATUS[$i]}" == "running" ]] && ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            wait "${PIDS[$i]}"; exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}  Env ${i} (PID: ${PIDS[$i]}): Finished successfully.${NC}"
                STATUS[$i]="finished"
            else
                echo -e "${RED}  Env ${i} (PID: ${PIDS[$i]}): Failed (code: ${exit_code}), restarting in ${RESTART_DELAY}s...${NC}"
                STATUS[$i]="restarting"
                RESTART_TIME[$i]=$(($(date +%s) + RESTART_DELAY))
            fi
        elif [[ "${STATUS[$i]}" == "restarting" ]] && [[ $(date +%s) -ge ${RESTART_TIME[$i]} ]]; then
            echo -e "${YELLOW}  Env ${i}: Restarting...${NC}"
            env_dir="${ENV_PREFIX}${i}"
            cmd=${CMDS[$i]}
            node_index=$(( (i-1) % NUM_NODES ))
            TARGET_NODE=${ALL_NODES[$node_index]}
            
            (
                log_file="env${i}.log"
                # exec确保ssh进程取代子shell，使得kill信号能正确传递
                exec ssh "$TARGET_NODE" "
                    source ${ENV_SCRIPT}
                    export HEAD_NODE_IP=${HEAD_NODE_IP}
                    cd '${PROJECT_ROOT}/${env_dir}'
                    mpiexec --mca btl_openib_allow_ib true -np ${NP} ${cmd}
                " >> "$log_file" 2>&1
            ) &
            
            PIDS[$i]=$!
            STATUS[$i]="running"
            # 更新PID地图文件
            sed -i "s/^\($i \)[0-9]*/\1${PIDS[$i]}/" "${PID_MAP_FILE}"
            echo -e "${GREEN}  ✓ Env ${i} restarted (ssh PID: ${PIDS[$i]}) on ${TARGET_NODE}${NC}"
        fi
    done
done

# --- 最终等待与总结 ---
echo "Waiting for any final processes to complete..."
wait
echo -e "\n${GREEN}🎉 All tasks completed. Exiting Slurm job.${NC}"
