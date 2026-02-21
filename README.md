# learn_repos

A consolidated collection of learning repositories covering various topics.

## Contents

| Directory | Topic |
|-----------|-------|
| `learn_html_css_js` | HTML, CSS, and JavaScript fundamentals |
| `learn_streamlit` | Streamlit app development |
| `learn_pandas` | Pandas data manipulation |
| `learn_ptl` | PyTorch Lightning training |
| `learn_cuda_programming` | CUDA GPU programming |
| `learn_Bentoml-samples` | BentoML model serving |
| `learn_sql_cs50` | SQL (CS50 course) |
| `learn_cs50` | CS50 general coursework |
| `learn_gradio_chat_engine` | Gradio chat interface |

## Setup

Most Python projects use [pixi](https://pixi.sh) for dependency management. To set up a project:

```bash
cd learn_<project>
pixi install
```

CUDA projects require `nvcc`:
```bash
cd learn_cuda_programming
nvcc vector_add.cu -o vector_add
```
