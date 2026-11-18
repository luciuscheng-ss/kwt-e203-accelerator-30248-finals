# 面向关键词识别的轻量化神经网络硬件加速器设计（30248，X11/W11/O12 1.07 决赛版）

- 公开工程仓库：`https://github.com/luciuscheng-ss/kwt-e203-accelerator-30248-finals`（唯一发行标签：`v20261118.1`）

本科组决赛必做第1—3项任务的完整技术工程。普通 MODE0，原 System Bus 配置1，完整 99×256 结果以原生 signed12 定长字段（每位置独占 12 位，行优先共 9,504 个 32 位字）写入共享 SRAM。自研 RTL、264 字节 CPU 驱动、验证工程、数值模型、参数转换脚本、冻结参数及配套答辩材料（`defense_materials/`）均在本工程中。

## 关键实测与综合结果
150 MHz；12 项冷任务各 52,700 周期（合计 632,400 周期），完整冷任务吞吐 2,846.30 task/s；全数字 SoC 平均功耗 31.51667 mW，平均单任务能耗 11.07285 µJ；保守新增面积 0.355243 mm²；本地 validation（每层 4,445 条）最差层 NRMSE 0.11517%，折算得分 9.61609/10；Setup 裕量 +0.000059985 ns，Hold 裕量 +0.000303321 ns。条件为 SS 0.99 V、125 ℃、pre-CTS 理想时钟。
最终网表 SHA256：33ec776dca68e9ce78fa6671fc5b3abf57f98eae7977b658746aa9081129ff87。

## 独立数值复现
```sh
git clone https://github.com/luciuscheng-ss/kwt-e203-accelerator-30248-finals.git
cd kwt-e203-accelerator-30248-finals
git checkout v20261118.1
python3 -m pip install -r requirements-numeric.txt
python3 tools/verify_release.py
python3 numeric/portable_convert.py encode --project project04 --input project04/examples/layer_00_input.npy --layer 0 --source-hex source0.hex --decode-metadata meta0.json
python3 project04/experiments/shared_bus_write_allocate_20261111/numeric_reference_native51.py --input project04/examples/layer_00_input.npy --layer 0 --reference-output reference0.npz
```

## 原平台 RTL 与网表验证
```sh
cd project04
python3 hardware/prepare.py --soc-root SOC_ROOT --vendors-record VENDOR_MODELS.json --out NEW_PREPARED
python3 verification/run_release.py --prepared NEW_PREPARED/source_manifest.json --soc SOC_ROOT --out NEW_RTL
python3 verification/run_release.py --prepared NEW_PREPARED/source_manifest.json --soc SOC_ROOT --mapping MAPPING_RESULT.json --out NEW_MAPPED
```
