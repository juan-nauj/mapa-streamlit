# Mapa

## 🛠️ System Requirements

- install [uv][uv]

## ⚙️ Setup

Install python 3.9 and create an environment using the following command:

```sh
uv python install 3.9
uv venv --python 3.9
```

Activate the environment:

```sh
.venv\Scripts\activate
```

Install the requirements:

```sh
uv pip install -r requirements.txt
```

> 💡 You need to create and install requirements only once. However, the environment must be activated every time you open a new terminal.

## 🚀 Running the App

Once the environment is activated, run the following command:

```sh
streamlit run app.py
```

[uv]: https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2