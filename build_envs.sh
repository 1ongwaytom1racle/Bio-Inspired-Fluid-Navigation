#!/bin/bash

# 环境数量
ENV_COUNT=5

# 资源文件所在目录
SOURCE_DIR="./start_files_np12"

# 记录当前工作目录
WORK_DIR=$(pwd)

echo "开始创建和编译所有环境..."
echo "源码目录: $WORK_DIR"
echo "资源文件目录: $SOURCE_DIR"

# --- 扫描参数组合文件夹 ---
echo ""
echo "=== 扫描参数组合文件夹 ==="
PARAM_DIRS=()
if [ -d "$SOURCE_DIR" ]; then
    cd "$SOURCE_DIR"
    for dir in */; do
        if [ -d "$dir" ]; then
            # 移除末尾的斜杠
            dir=${dir%/}
            PARAM_DIRS+=("$dir")
        fi
    done
    cd "$WORK_DIR"
    
    # 按名称排序参数组合文件夹
    if [ ${#PARAM_DIRS[@]} -gt 0 ]; then
        IFS=$'\n' PARAM_DIRS=($(sort <<<"${PARAM_DIRS[*]}"))
        unset IFS
        
        echo "找到 ${#PARAM_DIRS[@]} 个参数组合文件夹:"
        for i in "${!PARAM_DIRS[@]}"; do
            param_dir="${PARAM_DIRS[$i]}"
            echo "  $((i+1)). $param_dir"
            
            # 检查该参数组合下的文件
            if [ -d "$SOURCE_DIR/$param_dir" ]; then
                restart_count=$(find "$SOURCE_DIR/$param_dir" -maxdepth 1 -name "restore.*" -type d 2>/dev/null | wc -l)
                eel2d_count=$(find "$SOURCE_DIR/$param_dir" -maxdepth 1 -name "eel2d_*.vertex" -type f 2>/dev/null | wc -l)
                echo "     └─ 重启文件夹: ${restart_count}个, eel2d文件: ${eel2d_count}个"
            fi
        done
    else
        echo "未找到参数组合文件夹"
    fi
else
    echo "⚠️  资源目录 $SOURCE_DIR 不存在"
fi

# --- 创建分配日志文件 ---
LOG_FILE="$WORK_DIR/allocation_log.txt"
echo "=== 环境分配日志 ===" > "$LOG_FILE"
echo "生成时间: $(date)" >> "$LOG_FILE"
echo "工作目录: $WORK_DIR" >> "$LOG_FILE"
echo "源目录: $SOURCE_DIR" >> "$LOG_FILE"
echo "环境数量: $ENV_COUNT" >> "$LOG_FILE"
echo "参数组合数量: ${#PARAM_DIRS[@]}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

for i in $(seq 1 $ENV_COUNT); do
    echo ""
    echo "=== 处理环境 $i ==="
    
    # 创建环境目录
    echo "创建环境文件夹: $WORK_DIR/env$i"
    mkdir -p "$WORK_DIR/env$i"
    
    # 记录环境信息到日志
    echo "环境 $i:" >> "$LOG_FILE"
    
    # 检查是否有对应的参数组合
    if [ ${#PARAM_DIRS[@]} -ge $i ]; then
        param_dir="${PARAM_DIRS[$((i-1))]}"
        echo "分配参数组合: $param_dir"
        echo "  参数组合: $param_dir" >> "$LOG_FILE"
        
        # 扫描该参数组合下的重启文件夹
        if [ -d "$SOURCE_DIR/$param_dir" ]; then
            # 查找重启文件夹
            restart_dir=$(find "$SOURCE_DIR/$param_dir" -maxdepth 1 -name "restore.*" -type d | head -1)
            if [ -n "$restart_dir" ]; then
                restart_name=$(basename "$restart_dir")
                echo "复制重启文件夹: $param_dir/$restart_name -> env$i/restart_IB2dStrDiv/$restart_name"
                
                # 创建restart_IB2dStrDiv目录
                mkdir -p "$WORK_DIR/env$i/restart_IB2dStrDiv"
                
                # 复制重启文件夹
                cp -r "$restart_dir" "$WORK_DIR/env$i/restart_IB2dStrDiv/"
                if [ $? -eq 0 ]; then
                    echo "✓ 重启文件夹复制成功"
                    echo "    重启文件: $param_dir/$restart_name -> restart_IB2dStrDiv/$restart_name" >> "$LOG_FILE"
                else
                    echo "✗ 重启文件夹复制失败"
                    echo "    重启文件: 复制失败" >> "$LOG_FILE"
                fi
            else
                echo "⚠️  在 $param_dir 中未找到重启文件夹"
                echo "    重启文件: 未找到" >> "$LOG_FILE"
            fi
            
            # 查找eel2d文件
            eel2d_file=$(find "$SOURCE_DIR/$param_dir" -maxdepth 1 -name "eel2d_*.vertex" -type f | head -1)
            if [ -n "$eel2d_file" ]; then
                eel2d_name=$(basename "$eel2d_file")
                echo "复制eel2d文件: $param_dir/$eel2d_name -> env$i/eel2d.vertex"
                
                # 复制并重命名eel2d文件
                cp "$eel2d_file" "$WORK_DIR/env$i/eel2d.vertex"
                if [ $? -eq 0 ]; then
                    echo "✓ eel2d文件复制并重命名成功"
                    echo "    eel2d文件: $param_dir/$eel2d_name -> eel2d.vertex" >> "$LOG_FILE"
                else
                    echo "✗ eel2d文件复制失败"
                    echo "    eel2d文件: 复制失败" >> "$LOG_FILE"
                fi
            else
                echo "⚠️  在 $param_dir 中未找到eel2d文件"
                echo "    eel2d文件: 未找到" >> "$LOG_FILE"
            fi
        else
            echo "⚠️  参数组合文件夹 $SOURCE_DIR/$param_dir 不存在"
            echo "    参数组合: 文件夹不存在" >> "$LOG_FILE"
        fi
    else
        echo "无参数组合分配给环境 $i（从头开始）"
        echo "  参数组合: 从头开始" >> "$LOG_FILE"
    fi
    
    # 复制基础文件 input2d 和 cylinder2d.vertex
    echo "复制基础配置文件..."
    
    # 复制input2d文件
    if [ -f "$WORK_DIR/input2d" ]; then
        cp "$WORK_DIR/input2d" "$WORK_DIR/env$i/input2d"
        if [ $? -eq 0 ]; then
            echo "✓ input2d文件复制成功"
            echo "    input2d文件: ✓" >> "$LOG_FILE"
        else
            echo "✗ input2d文件复制失败"
            echo "    input2d文件: 复制失败" >> "$LOG_FILE"
        fi
    else
        echo "⚠️  当前目录未找到input2d文件"
        echo "    input2d文件: 未找到" >> "$LOG_FILE"
    fi
    
    # 复制cylinder2d.vertex文件
    if [ -f "$WORK_DIR/cylinder2d.vertex" ]; then
        cp "$WORK_DIR/cylinder2d.vertex" "$WORK_DIR/env$i/cylinder2d.vertex"
        if [ $? -eq 0 ]; then
            echo "✓ cylinder2d.vertex文件复制成功"
            echo "    cylinder2d.vertex文件: ✓" >> "$LOG_FILE"
        else
            echo "✗ cylinder2d.vertex文件复制失败"
            echo "    cylinder2d.vertex文件: 复制失败" >> "$LOG_FILE"
        fi
    else
        echo "⚠️  当前目录未找到cylinder2d.vertex文件"
        echo "    cylinder2d.vertex文件: 未找到" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    
    # 进入环境目录
    cd "$WORK_DIR/env$i"
    
    # 运行cmake
    echo "运行cmake..."
    cmake .. \
        -DCMAKE_C_COMPILER=/public1/soft/gcc/12.2/bin/gcc \
        -DCMAKE_CXX_COMPILER=/public1/soft/gcc/12.2/bin/g++ \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_PREFIX_PATH=/public21/home/sc93921/local_install/nlohmann_json
    
    # 检查cmake是否成功
    if [ $? -eq 0 ]; then
        echo "cmake成功，开始make..."
        make clean
        make -j2
        
        # 检查make是否成功
        if [ $? -eq 0 ]; then
            echo "✓ 环境 $i 创建和编译成功"
        else
            echo "✗ 环境 $i make失败"
        fi
    else
        echo "✗ 环境 $i cmake失败"
    fi
    
    # 回到工作目录
    mkdir CoupledStr
    cd "$WORK_DIR"
done

echo ""
echo "所有环境处理完成！"

# 显示结果汇总
echo ""
echo "=== 编译结果汇总 ==="
for i in $(seq 1 $ENV_COUNT); do
    if [ -f "$WORK_DIR/env$i/main2d" ]; then
        echo "✓ env$i: 编译成功"
        
        # 检查是否有重启文件
        if [ -d "$WORK_DIR/env$i/restart_IB2dStrDiv" ]; then
            restart_dirs=$(find "$WORK_DIR/env$i/restart_IB2dStrDiv" -name "restore.*" -type d 2>/dev/null | head -1)
            if [ -n "$restart_dirs" ]; then
                echo "  └─ 重启文件: $(basename "$restart_dirs")"
            else
                echo "  └─ 重启文件: restart_IB2dStrDiv/ (空)"
            fi
        else
            echo "  └─ 从头开始"
        fi
        
        # 检查是否有eel2d文件
        if [ -f "$WORK_DIR/env$i/eel2d.vertex" ]; then
            echo "  └─ eel2d文件: ✓"
        else
            echo "  └─ eel2d文件: ✗"
        fi
        
        # 检查是否有input2d文件
        if [ -f "$WORK_DIR/env$i/input2d" ]; then
            echo "  └─ input2d文件: ✓"
        else
            echo "  └─ input2d文件: ✗"
        fi
        
        # 检查是否有cylinder2d.vertex文件
        if [ -f "$WORK_DIR/env$i/cylinder2d.vertex" ]; then
            echo "  └─ cylinder2d.vertex文件: ✓"
        else
            echo "  └─ cylinder2d.vertex文件: ✗"
        fi
    else
        echo "✗ env$i: 编译失败或未找到可执行文件"
    fi
done

echo ""
echo "=== 参数组合分配汇总 ==="
for i in $(seq 1 $ENV_COUNT); do
    if [ ${#PARAM_DIRS[@]} -ge $i ]; then
        param_dir="${PARAM_DIRS[$((i-1))]}"
        echo "环境 $i: 参数组合 $param_dir"
        
        # 重启文件信息
        restart_dir=$(find "$WORK_DIR/env$i/restart_IB2dStrDiv" -name "restore.*" -type d 2>/dev/null | head -1)
        if [ -n "$restart_dir" ]; then
            echo "  重启文件: $(basename "$restart_dir")"
        else
            echo "  重启文件: 无"
        fi
        
        # eel2d文件信息
        if [ -f "$WORK_DIR/env$i/eel2d.vertex" ]; then
            echo "  eel2d文件: eel2d.vertex (已重命名)"
        else
            echo "  eel2d文件: 无"
        fi
        
        # input2d文件信息
        if [ -f "$WORK_DIR/env$i/input2d" ]; then
            echo "  input2d文件: ✓"
        else
            echo "  input2d文件: ✗"
        fi
        
        # cylinder2d.vertex文件信息
        if [ -f "$WORK_DIR/env$i/cylinder2d.vertex" ]; then
            echo "  cylinder2d.vertex文件: ✓"
        else
            echo "  cylinder2d.vertex文件: ✗"
        fi
    else
        echo "环境 $i: 从头开始"
    fi
done

# 显示日志文件位置
echo ""
echo "=== 分配日志 ==="
echo "详细分配信息已保存到: $LOG_FILE"
echo "可以使用以下命令查看完整日志:"
echo "  cat $LOG_FILE"

# 显示日志内容
echo ""
echo "=== 分配日志内容 ==="
cat "$LOG_FILE"
