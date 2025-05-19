
CONDA_PYTHON_PATH="/home/tju/miniconda3/envs/trg/bin/python"
PYTHON_SCRIPT_AND_ARGS="train.py --dataset_name enron --device cuda:0"
LOG_BASE_DIRECTORY="/home/tju/TRG/dynspec/logs"
LOG_FILENAME_PREFIX="dyspec"

CURRENT_DATE=$(date +'%Y%m%d')
DAILY_LOG_DIRECTORY="${LOG_BASE_DIRECTORY}/${CURRENT_DATE}"

mkdir -p "$DAILY_LOG_DIRECTORY"
if [ ! -d "$DAILY_LOG_DIRECTORY" ]; then
    echo "错误：无法创建或访问当天的日志子目录 '$DAILY_LOG_DIRECTORY'。"
    exit 1
fi
if [ ! -w "$DAILY_LOG_DIRECTORY" ]; then
    echo "错误：没有对当天的日志子目录 '$DAILY_LOG_DIRECTORY' 的写入权限。"
    exit 1
fi

CURRENT_TIME=$(date +'%H%M%S')
LOG_FILE="${DAILY_LOG_DIRECTORY}/${LOG_FILENAME_PREFIX}_${CURRENT_TIME}.log"

echo "----------------------------------------------------"
echo "准备启动命令..."
echo "日志将记录到: $LOG_FILE"
echo "----------------------------------------------------"

nohup "$CONDA_PYTHON_PATH" $PYTHON_SCRIPT_AND_ARGS > "$LOG_FILE" 2>&1 &

PROCESS_PID=$!
if [ $? -eq 0 ]; then
    echo "命令已成功在后台启动。"
    echo "PID: $PROCESS_PID"
else
    echo "错误：启动命令失败。"
fi
echo "----------------------------------------------------"