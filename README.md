# 股票分析系统

## 项目简介

这是一个专业的 A 股股票分析系统，提供全面的技术指标分析和投资建议。系统包括三个主要组件：

- 单股票分析
- 批量股票扫描器

## 功能特点

### 单股票分析

- 实时计算多种技术指标
- 生成详细的股票分析报告
- 提供投资建议
- 支持单股和批量分析

### 全市场扫描

- 扫描全部 A 股股票
- 根据多维度技术指标进行评分
- 筛选高潜力股票
- 按价格区间生成分析报告

## 技术指标

- 移动平均线 (Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 布林带 (Bollinger Bands)
- 能量潮指标 (OBV)
- 随机指标 (Stochastic Oscillator)
- 平均真实波动范围 (ATR)

## 系统依赖

- Python 3.12+
- Pandas
- NumPy
- AkShare
- Markdown2

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

#### 全市场股票扫描

```bash
python stock_scanner.py
```

分析结果将保存在 `data` 目录下：

- `high_score_stocks.xlsx`：按价格区间的详细分析

#### 个股 AI 分析

1. 配置

```
cp .env.example .env
```

- 在 `.env` 文件中配置硅基流动密钥

```
LLM_API_URL=https://api.siliconflow.cn
LLM_API_KEY=sk-xxxxxx
LLM_MODEL=THUDM/GLM-4-32B-0414
LLM_TIMEOUT=600
```

## 注意事项

- 股票分析仅供参考，不构成投资建议
- 使用前请确保网络连接正常
- 建议在实盘前充分测试

## 贡献

欢迎提交 issues 和 pull requests！

## 免责声明
本系统仅用于学习和研究目的，投资有风险，入市需谨慎。
