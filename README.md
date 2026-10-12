# 实验2 Label Studio 数据标注实践

用于在腾讯云 Cloud Studio 运行 Label Studio 和 ML backend，完成算术练习册的框选预标注与 OCR 文字精标。视觉/OCR 推理调用外部兼容接口；只运行标注服务时可使用 CPU 环境。

仓库只包含代码、启动配置和使用说明。教师报告、课程图片、参考答案、登录账号、数据库和模型密钥均不纳入 Git；课程数据另行上传到工作空间。

## 从 Git 仓库创建应用

在“创建应用 → 从 Git 仓库导入”填写：

- 仓库地址：`https://github.com/XLhuanbin/cloudstudio-labelstudio-lab2.git`。
- 应用标题：`实验2 数据标注实践`。
- 应用描述：`基于 Label Studio 和 ML backend 的算术练习册半自动标注，支持视觉模型预标注和 OCR 文字精标。`

当前页面注明暂不支持私有仓库。代码拉取后进入应用的工作空间，选择 Python 3.10 或 3.11。Git 导入并不表示服务已启动；在工作空间终端执行下列步骤。如后续页面提供启动命令和服务端口，可填 `bash start.sh` 和 `8080`，并先完成环境安装。

## 首次准备与启动

在仓库根目录上传并解压随附的 `实验2数据_CloudStudio上传.zip`，解压后应有 `实验2数据/train/annotations.json`、`实验2数据/train/images/`、`实验2数据/同类测试/images/` 和 `实验2数据/鲁棒测试/images/`，不要多套一层仓库目录。

```bash
bash setup.sh
cp .env.example .env
# 在编辑器中配置 .env；没有模型接口也能先运行标注软件。
bash start.sh
```

若工作空间自带的 `python3` 不是 Python 3.10/3.11，可用 `PYTHON_BIN=/实际/python路径 bash setup.sh` 指定解释器。依赖固定了本机验证过的关键版本，Linux 环境的最终安装情况以 Cloud Studio 的 `pip check` 为准。

启动器会生成并检查 `.runtime/tasks.json`，随后同时启动 8080 的 Label Studio 和只监听内部 9090 的 ML backend。只需在 Cloud Studio 的端口预览中打开 **8080**。首次打开 Label Studio 注册账号，或提前填写 `.env` 中的 `LABEL_STUDIO_USERNAME` 和 `LABEL_STUDIO_PASSWORD`。

服务进程和数据库保存在当前工作空间；停止工作空间后服务会停止。数据库和导出结果需要按平台的持久化方式保留。

## 导入120个任务并连接后端

登录 Label Studio，在 Account & Settings 复制 API Token，填入工作空间 `.env` 的 `LABEL_STUDIO_API_KEY`。在另一个终端执行：

```bash
set -a
source .env
set +a
.venv/bin/python import_project.py --connect-backend
```

脚本创建“实验2 数据标注实践”项目，写入标注界面，授权课程图片目录，并导入训练集100张、同类测试10张、鲁棒测试10张。重复运行会按图片路径跳过已有任务。ML backend 使用同一工作空间内部的 `http://127.0.0.1:9090`，不需要把9090作为公开预览端口。

如果选择手动操作，先用 `配置/label_config.xml` 创建项目；在 Settings → Cloud Storage 添加 Local files，Path 填工作空间中 `实验2数据` 的绝对路径，只授权读取，不同步扫描；然后导入 `.runtime/tasks.json`。在 Settings → Machine Learning 添加上面的内部后端地址。

## 配置视觉模型与OCR

编辑 `.env` 中的 `QWEN_VL_API_URL`、`QWEN_VL_API_KEY`、`QWEN_VL_MODEL`、`OCR_API_URL`、`OCR_API_KEY` 和 `OCR_MODEL`，停止并重新运行 `bash start.sh`。URL 是完整的 `/chat/completions` 请求地址；模型名称应与服务商实际开放的名称一致。

两个接口都需要支持 OpenAI-compatible 的图片消息请求及 `choices[0].message.content` 响应。不同供应商的原生 PaddleOCR/HunyuanOCR 接口可能采用不同协议，届时需根据实际接口调整适配器。视觉响应内容是 JSON 数组：

```json
[{"x": 5, "y": 10, "width": 30, "height": 5, "text": "50+40=90", "score": 0.9}]
```

坐标相对原图，以0到100的百分比表示。OCR 接收算式框裁剪图，返回算式文本，不应替学生纠正答案。未配置视觉端点时，健康检查仍可通过，但预测会明确返回未配置错误，不能据此声称完成模型推理。首次只选一张图验证，再批量执行，尤其是逐框调用 OCR 时。

## 导出和评估

训练集导入的2974个框、2850条文本是随附参考标注。两个测试集不含参考答案；若要计算测试指标，先建立独立人工参考标注，保留原始预测再比对。未经独立人工标注不能计算这两组的召回率或OCR准确率。

```bash
.venv/bin/python 源码/evaluate_annotations.py \
  --tasks .runtime/tasks.json \
  --predictions 导出的预测.json \
  --report .runtime/prediction_metrics.json
```

现有评估脚本计算 IoU≥0.5 的精确率/召回率、匹配框平均IoU和匹配文本的整式完全一致率，不计算逐字符准确率。只有 `ground_truth` 参考任务参与评估；默认任务文件只能评估含独立预测的训练图片。

## 排错

- 页面打不开：检查8080预览、工作空间是否运行，以及 `.runtime/labelstudio.log`。
- 图片403/404：确认解压层级、Local files目录授权及 `.runtime/task_check.json`。
- 预览重定向或静态资源地址错误：把 `.env` 的 `LABEL_STUDIO_HOST` 填为当前实际 HTTPS 预览根地址，重启服务。
- 预测失败：先查看 `.runtime/ml_backend.log`，确认实际模型名、完整接口地址和接口协议。
- 端口占用：停止旧进程，或修改 `.env` 的 `APP_PORT` / `ML_BACKEND_PORT`；预览端口和后端连接地址随之调整。

`.env`、`.runtime/` 和 `实验2数据/` 已加入 `.gitignore`。发布前检查 `git status`，不要强制提交这些文件。
