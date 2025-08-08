import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import logging
import traceback
import matplotlib

matplotlib.use("Agg")  # Use um backend não interativo
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass

# --- Configuração da Página e Logging ---
st.set_page_config(layout="wide", page_title="Aeromap - TRX", initial_sidebar_state="collapsed")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Classes de Configuração e Constantes ---


@dataclass
class AeroConstants:
    """Constantes aerodinâmicas e ambientais."""

    AREA_FIXA: float = 1.05
    DENSIDADE_FIXA: float = 1.225
    CORRECAO_CLA: float = 0.1


@dataclass
class VehicleParams:
    """Parâmetros físicos do modelo de bicicleta."""

    G: float = 9.81
    MASSA: float = 287.0
    INERCIA_YAW: float = 92.48
    DIST_EIXOS: float = 1.55
    A_CG: float = DIST_EIXOS * (51.34 / 100)
    B_CG: float = DIST_EIXOS - A_CG
    SR_RATIO: float = 0.17  # Razão de esterçamento


@dataclass
class TireModelParams:
    """Parâmetros do modelo de pneu Pacejka Magic Formula (Hoosier R20)."""

    FZ0: float = 1080.0
    P0: float = 96526.6
    Pi: float = 103421.0
    PCY1: float = 1.5
    PDY1: float = 2.2598
    PDY2: float = -0.43603
    PDY3: float = 8.2095
    PEY1: float = -0.037535
    PEY2: float = 0.0099506
    PEY3: float = -0.70019
    PEY4: float = -142.1786
    PKY1: float = -25.4905
    PKY2: float = 1.3365
    PKY3: float = 0.57413
    PHY1: float = -0.0028777
    PHY2: float = -0.00043311
    PHY3: float = 0.032931
    PVY1: float = -0.054319
    PVY2: float = 0.035064
    PVY3: float = 1.1708
    PVY4: float = 0.15011
    PPY1: float = 0.52612
    PPY2: float = 1.4789
    PPY3: float = -0.43266
    PPY4: float = -0.67053
    LKY: float = 1
    LEY: float = 1
    LFZ0: float = 1
    LGAZ: float = 1
    LHY: float = 1
    LMUY: float = 1
    LVY: float = 1
    LCY: float = 1


# --- Nomes de Features e Labels ---
FEATURES_ENTRADA = ["vel", "FRH", "RRH", "vol_pos", "AoD", "AoT"]
FEATURES_SAIDA_MODELO = ["ClA", "CdA", "Cpz", "DD", "DE", "m"]
FEATURES_SAIDA_DERIVADAS = ["Eficiencia", "Balanco_Aero"]
FEATURES_SAIDA_TODAS = FEATURES_SAIDA_MODELO + FEATURES_SAIDA_DERIVADAS

LABELS_ENTRADA = {
    "vel": "Velocidade (km/h)",
    "FRH": "Altura Dianteira (mm)",
    "RRH": "Altura Traseira (mm)",
    "vol_pos": "Ângulo Volante (°)",
    "AoD": "AoA Asa Diant. (°)",
    "AoT": "AoA Asa Tras. (°)",
}
LABELS_SAIDA = {
    "ClA": "ClA",
    "CdA": "CdA",
    "Cpz": "Cp Longitudinal (mm)",
    "DD": "Fluxo Duto Dir. (g/s)",
    "DE": "Fluxo Duto Esq. (g/s)",
    "m": "Fluxo Radiador (g/s)",
    "Eficiencia": "Eficiência Aero (Cl/Cd)",
    "Balanco_Aero": "Balanço Aero (%)",
    "Raio_Curva": "Raio de Curva (m)",
}


# --- Caching: Carregar modelos de ML apenas uma vez ---
@st.cache_resource
def load_models_and_scaler():
    """Carrega o scaler e o ensemble de modelos Keras."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_y_path = os.path.join(script_dir, "scaler_y.pkl")
        scaler_y = joblib.load(scaler_y_path)

        models_path = os.path.join(script_dir, "models")
        model_files = [f for f in os.listdir(models_path) if f.endswith(".keras")]
        if not model_files:
            st.error("Nenhum modelo encontrado no diretório 'models'.")
            return None, None
        models_ensemble = [tf.keras.models.load_model(os.path.join(models_path, f)) for f in sorted(model_files)]
        return scaler_y, models_ensemble
    except Exception as e:
        st.error(f"ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
        traceback.print_exc()
        return None, None


scaler_y, models_ensemble = load_models_and_scaler()


# --- Funções de Lógica Principal ---
def get_aero_coefficients(input_df: pd.DataFrame) -> pd.DataFrame:
    """Executa a predição do ensemble e calcula as saídas derivadas."""
    if not models_ensemble or not scaler_y:
        raise RuntimeError("Erro de Servidor: Modelos ou scaler não disponíveis.")

    # Predição
    preds_scaled = [model.predict(input_df[FEATURES_ENTRADA].values, verbose=0) for model in models_ensemble]
    mean_pred_scaled = np.mean(preds_scaled, axis=0)
    pred_final = scaler_y.inverse_transform(mean_pred_scaled)
    df_pred = pd.DataFrame(pred_final, columns=FEATURES_SAIDA_MODELO, index=input_df.index)
    df_pred["ClA"] += AeroConstants.CORRECAO_CLA

    # Cálculos Derivados
    df_pred["Eficiencia"] = np.divide(
        df_pred["ClA"], df_pred["CdA"], out=np.zeros_like(df_pred["ClA"]), where=(df_pred["CdA"] > 0)
    )
    df_pred["Balanco_Aero"] = (1 - ((775 - df_pred["Cpz"]) / 1550)) * 100

    return df_pred


def calculate_magic_formula_force(alpha: float, Fz: float, tire_params: TireModelParams, gamma_rad: float = 0) -> float:
    """
    Fórmula Mágica de Pacejka para força lateral do pneu, acessando os parâmetros
    diretamente do objeto dataclass, sem conversão para dicionário.
    """
    if Fz <= 0:
        return 0.0

    # Acessando cada parâmetro diretamente do objeto 'tire_params'
    dfz = (Fz - tire_params.FZ0) / tire_params.FZ0
    dpi = (tire_params.Pi - tire_params.P0) / tire_params.P0
    gamma_y = gamma_rad * tire_params.LGAZ

    C = tire_params.PCY1 * tire_params.LCY
    mu_y = (
        (tire_params.PDY1 + tire_params.PDY2 * dfz)
        * (1 + tire_params.PPY3 * dpi + tire_params.PPY4 * dpi**2)
        * (1 - tire_params.PDY3 * gamma_y**2)
    )
    D = mu_y * Fz * tire_params.LMUY

    K_yfz = (
        tire_params.PKY1
        * tire_params.FZ0
        * (1 + tire_params.PPY1 * dpi)
        * np.sin(2 * np.arctan(Fz / (tire_params.PKY2 * tire_params.FZ0 * (1 + tire_params.PPY2 * dpi))))
    )
    K = K_yfz * tire_params.LFZ0 * tire_params.LKY * (1 - tire_params.PKY3 * abs(gamma_y))

    if abs(C * D) < 1e-6:
        return 0.0

    B = K / (C * D)
    E = (
        (tire_params.PEY1 + tire_params.PEY2 * dfz)
        * (1 - (tire_params.PEY3 + tire_params.PEY4 * gamma_y))
        * tire_params.LEY
    )

    SHy = (tire_params.PHY1 + tire_params.PHY2 * dfz) * tire_params.LHY + tire_params.PHY3 * gamma_y
    SVy = (
        Fz * (tire_params.PVY1 + tire_params.PVY2 * dfz) * tire_params.LVY * tire_params.LMUY
        + Fz * (tire_params.PVY3 + tire_params.PVY4 * dfz) * gamma_y * tire_params.LMUY
    )

    alpha_shifted = alpha + SHy
    arg_atan = B * alpha_shifted - E * (B * alpha_shifted - np.arctan(B * alpha_shifted))

    Fy_raw = D * np.sin(C * np.arctan(arg_atan)) + SVy
    return -Fy_raw


def vehicle_dynamics_ode(t, state, Vx, delta_roda, Fz_f, Fz_r, vehicle: VehicleParams, tire: TireModelParams):
    Vy, r = state
    if Vx < 0.1:
        return [0, 0]
    alpha_f = delta_roda - np.arctan2(Vy + vehicle.A_CG * r, Vx)
    alpha_r = -np.arctan2(Vy - vehicle.B_CG * r, Vx)
    Fyf = calculate_magic_formula_force(alpha_f, Fz_f, tire)
    Fyr = calculate_magic_formula_force(alpha_r, Fz_r, tire)
    ay = (Fyf + Fyr) / vehicle.MASSA
    dr_dt = (vehicle.A_CG * Fyf - vehicle.B_CG * Fyr) / vehicle.INERCIA_YAW
    dVy_dt = ay - Vx * r
    return [dVy_dt, dr_dt]


def calculate_simulated_radius(input_data: dict) -> tuple[float, pd.DataFrame]:
    df_input_single = pd.DataFrame([input_data], columns=FEATURES_ENTRADA)
    df_coeffs = get_aero_coefficients(df_input_single)
    Vx_ms = float(input_data["vel"]) / 3.6
    vehicle, tire, aero = VehicleParams(), TireModelParams(), AeroConstants()
    delta_roda_rad = np.deg2rad(float(input_data["vol_pos"])) * vehicle.SR_RATIO
    cla = df_coeffs["ClA"].iloc[0]
    balanco_aero = df_coeffs["Balanco_Aero"].iloc[0] / 100.0
    downforce_total = 0.5 * aero.DENSIDADE_FIXA * (Vx_ms**2) * aero.AREA_FIXA * cla
    downforce_f = downforce_total * balanco_aero
    downforce_r = downforce_total * (1 - balanco_aero)
    Fz_f_total = (vehicle.MASSA * vehicle.G * vehicle.B_CG / vehicle.DIST_EIXOS) + downforce_f
    Fz_r_total = (vehicle.MASSA * vehicle.G * vehicle.A_CG / vehicle.DIST_EIXOS) + downforce_r
    sol = solve_ivp(
        fun=vehicle_dynamics_ode,
        t_span=[0, 5],
        y0=[0, 0],
        method="RK45",
        args=(Vx_ms, delta_roda_rad, Fz_f_total, Fz_r_total, vehicle, tire),
        dense_output=True,
    )
    Vy_final, r_final = sol.y[:, -1]
    derivadas_finais = vehicle_dynamics_ode(
        0, [Vy_final, r_final], Vx_ms, delta_roda_rad, Fz_f_total, Fz_r_total, vehicle, tire
    )
    ay = derivadas_finais[0] + Vx_ms * r_final
    raio_curva = (Vx_ms**2) / abs(ay) if abs(ay) > 1e-6 else float("inf")
    return raio_curva, df_coeffs


# --- Funções de UI e Plotagem ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", decimal=".").encode("utf-8")


def generate_predictive_data(
    var_x: str, x_range: list, var_y: str, y_range: list, constantes: dict, resolucao: int
) -> pd.DataFrame:
    is_3d = var_y is not None
    x_vals = np.linspace(x_range[0], x_range[1], resolucao)
    if is_3d:
        y_vals = np.linspace(y_range[0], y_range[1], resolucao)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_dict = {**constantes, var_x: xx.flatten(), var_y: yy.flatten()}
    else:
        grid_dict = {**constantes, var_x: x_vals}
    df_input_grid = pd.DataFrame(grid_dict)
    df_final_grid = get_aero_coefficients(df_input_grid)
    return pd.concat([df_input_grid, df_final_grid], axis=1)


def create_plotly_figure(df_plot: pd.DataFrame, var_x: str, var_y: str, var_z: str, plot_type: str) -> go.Figure:
    is_3d = var_y is not None
    fig = go.Figure()
    x_label, z_label = LABELS_ENTRADA.get(var_x, var_x), LABELS_SAIDA.get(var_z, var_z)
    colorscale = "Jet"
    if is_3d:
        y_label = LABELS_ENTRADA.get(var_y, var_y)
        resolucao = int(np.sqrt(len(df_plot)))
        x_vals, y_vals = df_plot[var_x].unique(), df_plot[var_y].unique()
        z_data = df_plot[var_z].values.reshape(resolucao, resolucao)
        trace_params = dict(x=x_vals, y=y_vals, z=z_data, colorscale=colorscale, colorbar_title_text=z_label)
        if plot_type == "Superfície":
            fig.add_trace(go.Surface(**trace_params, name="Mapa Preditivo", opacity=0.8))
            fig.update_layout(
                title=f"Mapa Preditivo: {z_label}",
                scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
                height=800,
            )
        else:
            fig.add_trace(
                go.Contour(**trace_params, name="Mapa Preditivo", connectgaps=True, contours=dict(coloring="heatmap"))
            )
            fig.update_layout(title=f"Mapa Preditivo: {z_label}", xaxis_title=x_label, yaxis_title=y_label, height=700)
    else:
        fig.add_trace(go.Scatter(x=df_plot[var_x], y=df_plot[var_z], mode="lines"))
        fig.update_layout(
            title=f"Análise Preditiva: {z_label} vs {x_label}", xaxis_title=x_label, yaxis_title=z_label, height=600
        )
    return fig


# --- Aplicação Principal ---
st.title("Aeromap - TRX 🏎️")

if scaler_y is None or models_ensemble is None:
    st.error("Aplicação não pode iniciar pois os modelos falharam ao carregar. Verifique os logs.")
else:
    tabs = st.tabs(["Análise Singular", "Análise Preditiva", "Análise Dinâmica", "Análise de Dados Reais"])

    # --- ABA 1: Análise Singular ---
    with tabs[0]:
        # ... (código da Aba 1 permanece o mesmo)
        st.markdown(
            "Esta aba permite uma predição ponto a ponto. Insira um conjunto de parâmetros para calcular os coeficientes aerodinâmicos e o raio de curva simulado."
        )
        st.subheader("Predição Pontual")

        with st.form(key="prediction_form"):
            cols = st.columns(3)
            inputs = {}
            default_vals = {"vel": 36.39, "FRH": 40.32, "RRH": 47.62, "vol_pos": -12.75, "AoD": 0.99, "AoT": 0.75}
            for i, (key, label) in enumerate(LABELS_ENTRADA.items()):
                inputs[key] = cols[i % 3].number_input(label, value=default_vals[key], step=0.01)

            if st.form_submit_button("Prever Coeficientes e Raio"):
                with st.spinner("Calculando..."):
                    try:
                        radius, coeffs = calculate_simulated_radius(inputs)
                        st.subheader("Resultados da Predição")
                        result_dict = {name: f"{coeffs[name].iloc[0]:.4f}" for name in FEATURES_SAIDA_TODAS}
                        result_dict["Raio_Curva"] = f"{radius:.2f}"
                        res_cols = st.columns(3)
                        for i, key in enumerate(result_dict.keys()):
                            with res_cols[i % 3]:
                                st.metric(label=LABELS_SAIDA.get(key, key), value=result_dict[key])
                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a predição: {e}")
                        traceback.print_exc()

    # --- ABA 2: Análise Preditiva ---
    with tabs[1]:
        # ... (código da Aba 2 permanece o mesmo)
        st.markdown(
            "Esta aba gera mapas preditivos. Crie gráficos 2D ou 3D para visualizar como uma variável de saída muda em resposta a uma ou duas variáveis de entrada."
        )
        with st.form("predictive_graph_form"):
            analysis_type = st.radio("Tipo de Análise:", ("3D", "2D"), horizontal=True, key="pred_type")
            is_3d = analysis_type == "3D"
            main_cols = st.columns([2, 1])
            with main_cols[0]:
                st.subheader("Variáveis Independentes (Eixos)")
                axis_cols = st.columns(2)
                var_x_pred = axis_cols[0].selectbox(
                    "Eixo X:", options=FEATURES_ENTRADA, format_func=LABELS_ENTRADA.get, index=1
                )
                x_min_pred = axis_cols[0].number_input("Min X:", value=40.0)
                x_max_pred = axis_cols[0].number_input("Max X:", value=60.0)
                var_y_pred = None
                if is_3d:
                    options_y = [v for v in FEATURES_ENTRADA if v != var_x_pred]
                    var_y_pred = axis_cols[1].selectbox(
                        "Eixo Y:", options=options_y, format_func=LABELS_ENTRADA.get, index=1
                    )
                    y_min_pred = axis_cols[1].number_input("Min Y:", value=35.0)
                    y_max_pred = axis_cols[1].number_input("Max Y:", value=55.0)
            with main_cols[1]:
                st.subheader("Variável de Saída e Constantes")
                output_options = {k: v for k, v in LABELS_SAIDA.items() if k != "Raio_Curva"}
                var_z_pred = st.selectbox(
                    "Variável de Saída:", options=output_options.keys(), format_func=output_options.get
                )
                st.markdown("###### Valores Constantes")
                constantes = {}
                selected_axes = [var_x_pred, var_y_pred]
                for key, label in LABELS_ENTRADA.items():
                    if key not in selected_axes:
                        constantes[key] = st.number_input(
                            f"Constante {label}", value=36.39 if key == "vel" else 0.0, step=0.1, key=f"const_{key}"
                        )
            bottom_cols = st.columns(2)
            resolucao_pred = bottom_cols[0].slider("Resolução (Pontos):", 10, 100, 40)
            plot_type_3d = (
                bottom_cols[1].radio("Visualização 3D:", ("Contorno", "Superfície"), horizontal=True) if is_3d else None
            )
            if st.form_submit_button("Gerar Gráfico Preditivo"):
                if x_min_pred >= x_max_pred or (is_3d and y_min_pred >= y_max_pred):
                    st.error("Valor mínimo deve ser menor que o máximo.")
                else:
                    with st.spinner("Gerando dados para o gráfico..."):
                        try:
                            df_plot = generate_predictive_data(
                                var_x=var_x_pred,
                                x_range=[x_min_pred, x_max_pred],
                                var_y=var_y_pred,
                                y_range=[y_min_pred, y_max_pred] if is_3d else None,
                                constantes=constantes,
                                resolucao=resolucao_pred,
                            )
                            st.session_state.predictive_df = df_plot
                            st.session_state.plot_params = (
                                var_x_pred,
                                var_y_pred,
                                var_z_pred,
                                plot_type_3d if is_3d else "2D",
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar dados: {e}")
                            traceback.print_exc()
        if "predictive_df" in st.session_state:
            df_plot = st.session_state.predictive_df
            var_x, var_y, var_z, plot_type = st.session_state.plot_params
            fig = create_plotly_figure(df_plot, var_x, var_y, var_z, plot_type)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download Dados do Gráfico (.csv)", convert_df_to_csv(df_plot), "predictive_plot_data.csv", "text/csv"
            )

    # --- ABA 3: Análise Dinâmica (REINTEGRADA) ---
    with tabs[2]:
        st.markdown(
            "Realize uma análise dinâmica de uma volta completa. Carregue um arquivo de log, defina os parâmetros do veículo e visualize os pontos de operação sobre um mapa aerodinâmico."
        )
        st.subheader("Análise Dinâmica de Volta Completa")

        log_file = st.file_uploader("Carregue seu arquivo de log (.csv)", type="csv", key="dynamic_log_uploader")
        if st.button("Carregar Log de Exemplo"):
            try:
                example_path = os.path.join(os.path.dirname(__file__), "Ax_Ay_mpa.csv")
                st.session_state.log_df = pd.read_csv(example_path, sep=",", decimal=".", on_bad_lines="skip")
                st.success("Log de exemplo carregado!")
            except Exception as e:
                st.error(f"Erro ao carregar log de exemplo: {e}")

        if log_file:
            st.session_state.log_df = pd.read_csv(log_file, sep=",", decimal=".", on_bad_lines="skip")
            st.success(f"Arquivo '{log_file.name}' carregado!")

        if "log_df" in st.session_state:
            st.divider()
            st.markdown("##### Defina os Parâmetros da Análise")
            with st.form("dynamic_analysis_form"):
                p = {}  # Dicionário de parâmetros
                with st.expander("Parâmetros Físicos do Veículo", expanded=True):
                    cols = st.columns(3)
                    p["mass"] = cols[0].number_input("Massa (kg):", value=290)
                    p["cgy"] = cols[1].number_input("Altura CG (mm):", value=307)
                    p["wheelbase"] = cols[2].number_input("Entre-eixos (mm):", value=1550)
                    p["ride_rate_f"] = cols[0].number_input(
                        "Constante Elástica Diant. (N/mm):", value=63.67, format="%.2f"
                    )
                    p["ride_rate_r"] = cols[1].number_input(
                        "Constante Elástica Tras. (N/mm):", value=58.86, format="%.2f"
                    )

                with st.expander("Parâmetros de Setup do Veículo", expanded=True):
                    cols = st.columns(3)
                    p["static_frh"] = cols[0].number_input("Altura Estática Diant. (mm):", value=45)
                    p["static_rrh"] = cols[1].number_input("Altura Estática Tras. (mm):", value=45)
                    p["area"] = cols[2].number_input("Área Frontal (m²):", value=1.05, format="%.2f")
                    p["aero_balance"] = cols[0].number_input("Balanço Aero Dianteiro (%):", value=60.0) / 100.0
                    p["cla"] = cols[1].number_input(
                        "Coef. de Sustentação (ClA) Fixo:",
                        value=3.0,
                        help="ClA geral usado para cálculo de downforce neste modelo.",
                    )

                with st.expander("Parâmetros do Mapa Aero de Fundo", expanded=True):
                    cols = st.columns(2)
                    constant_inputs = {}
                    constant_inputs["AoD"] = cols[0].number_input(LABELS_ENTRADA["AoD"], value=0.0, key="dyn_aod")
                    constant_inputs["AoT"] = cols[1].number_input(LABELS_ENTRADA["AoT"], value=0.0, key="dyn_aot")

                st.divider()
                st.markdown("##### Selecione a Saída e Visualização")
                cols = st.columns(2)
                output_options_dyn = {k: v for k, v in LABELS_SAIDA.items() if k != "Raio_Curva"}
                dyn_var_z = cols[0].selectbox(
                    "Saída para o Mapa de Fundo:", options=output_options_dyn.keys(), format_func=output_options_dyn.get
                )
                dyn_plot_type = cols[1].radio("Tipo de Visualização:", ("Contorno", "Superfície"), horizontal=True)

                if st.form_submit_button("Gerar Análise Dinâmica"):
                    with st.spinner("Executando análise dinâmica..."):
                        try:
                            log_df = st.session_state.log_df
                            expected_cols = ["TIME", "WPS", "Velocidade_de_referência", "Força_G_aceleração"]
                            if not all(col in log_df.columns for col in expected_cols):
                                st.error(f"Arquivo de log inválido. Colunas esperadas: {expected_cols}")
                            else:
                                # Cálculos de dinâmica
                                g = 9.81
                                log_ax = log_df["Força_G_aceleração"] * g
                                speed_ms = log_df["Velocidade_de_referência"] / 3.6
                                long_wt = (p["cgy"] * p["mass"] * log_ax) / p["wheelbase"]
                                downforce_f = (
                                    0.5
                                    * AeroConstants.DENSIDADE_FIXA
                                    * p["cla"]
                                    * p["area"]
                                    * (speed_ms**2)
                                    * p["aero_balance"]
                                )
                                downforce_r = (
                                    0.5
                                    * AeroConstants.DENSIDADE_FIXA
                                    * p["cla"]
                                    * p["area"]
                                    * (speed_ms**2)
                                    * (1 - p["aero_balance"])
                                )
                                w_f_elast, w_r_elast = (downforce_f / 2) - long_wt, (downforce_r / 2) + long_wt
                                delta_frh, delta_rrh = w_f_elast / p["ride_rate_f"], w_r_elast / p["ride_rate_r"]
                                frh_dynamic, rrh_dynamic = (p["static_frh"] + delta_frh), (p["static_rrh"] + delta_rrh)

                                # Gerar mapa de fundo usando a função reutilizável
                                avg_speed_kmh = log_df["Velocidade_de_referência"].mean()
                                const_for_grid = {
                                    "vel": avg_speed_kmh,
                                    "vol_pos": log_df["WPS"].mean(),
                                    **constant_inputs,
                                }
                                df_mapa_fundo = generate_predictive_data(
                                    var_x="FRH",
                                    x_range=[frh_dynamic.min(), frh_dynamic.max()],
                                    var_y="RRH",
                                    y_range=[rrh_dynamic.min(), rrh_dynamic.max()],
                                    constantes=const_for_grid,
                                    resolucao=25,
                                )

                                # Criar a figura base (mapa)
                                fig = create_plotly_figure(df_mapa_fundo, "FRH", "RRH", dyn_var_z, dyn_plot_type)

                                # Adicionar os pontos da volta (scatter plot)
                                sample_rate = max(1, len(frh_dynamic) // 1000)
                                frh_sampled = frh_dynamic[::sample_rate]
                                rrh_sampled = rrh_dynamic[::sample_rate]

                                if dyn_plot_type == "Superfície":
                                    # Para plot 3D, precisamos do valor Z dos pontos da volta
                                    df_scatter_input = pd.DataFrame(
                                        {**const_for_grid, "FRH": frh_sampled, "RRH": rrh_sampled}
                                    )
                                    scatter_z_values = get_aero_coefficients(df_scatter_input)[dyn_var_z]
                                    fig.add_trace(
                                        go.Scatter3d(
                                            x=frh_sampled,
                                            y=rrh_sampled,
                                            z=scatter_z_values,
                                            mode="markers",
                                            name="Pontos da Volta",
                                            marker=dict(color="black", size=3, opacity=0.8),
                                        )
                                    )
                                else:  # Contorno
                                    fig.add_trace(
                                        go.Scatter(
                                            x=frh_sampled,
                                            y=rrh_sampled,
                                            mode="markers",
                                            name="Pontos da Volta",
                                            marker=dict(color="black", size=6, opacity=0.7),
                                        )
                                    )

                                fig.update_layout(
                                    title="Análise Dinâmica: Pontos de Operação vs. Mapa Preditivo",
                                    xaxis_title="FRH Dinâmico (mm)",
                                    yaxis_title="RRH Dinâmico (mm)",
                                )

                                st.info(
                                    f"Mapa de fundo gerado com a velocidade média da volta: {avg_speed_kmh:.2f} km/h"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Armazenar para download
                                st.session_state.dynamic_results_df = pd.DataFrame(
                                    {"dynamic_frh_mm": frh_dynamic, "dynamic_rrh_mm": rrh_dynamic}
                                )

                        except Exception as e:
                            st.error(f"Ocorreu um erro durante a análise dinâmica: {e}")
                            traceback.print_exc()

            if "dynamic_results_df" in st.session_state:
                st.download_button(
                    "Download Dados Dinâmicos da Volta (.csv)",
                    convert_df_to_csv(st.session_state.dynamic_results_df),
                    "dynamic_analysis_lap_data.csv",
                    "text/csv",
                )

    # --- ABA 4: Análise de Dados Reais ---
    with tabs[3]:
        # ... (código da Aba 4 permanece o mesmo)
        st.markdown(
            "Valide a precisão do modelo. Carregue um CSV com dados de entrada e saídas reais (medidas). A ferramenta gerará gráficos de correlação comparando os dados reais com as predições."
        )
        st.subheader("Análise de Correlação: Real vs. Previsto")
        real_data_file = st.file_uploader(
            "Carregue seu arquivo de dados reais (.csv)", type="csv", key="real_data_uploader"
        )
        if st.button("Carregar Dados de Exemplo", key="ex_real"):
            try:
                example_path = os.path.join(os.path.dirname(__file__), "dados_reais_e_simulados.csv")
                st.session_state.real_df_to_process = pd.read_csv(example_path, sep=";", decimal=".")
                st.success("Arquivo de dados reais de exemplo carregado!")
            except Exception as e:
                st.error(f"Erro ao carregar arquivo de exemplo: {e}")
        if real_data_file:
            st.session_state.real_df_to_process = pd.read_csv(real_data_file, sep=";", decimal=".")
        if "real_df_to_process" in st.session_state:
            with st.spinner("Gerando gráficos de correlação..."):
                df_real = st.session_state.real_df_to_process
                required_cols = FEATURES_ENTRADA + FEATURES_SAIDA_MODELO
                if not all(col in df_real.columns for col in required_cols):
                    st.error(f"Colunas ausentes no arquivo: {set(required_cols) - set(df_real.columns)}")
                else:
                    df_processed = df_real[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
                    if df_processed.empty:
                        st.error("Nenhum dado válido encontrado após a limpeza.")
                    else:
                        input_df = df_processed[FEATURES_ENTRADA]
                        real_output_df = df_processed[FEATURES_SAIDA_MODELO]
                        df_predito = get_aero_coefficients(input_df)[FEATURES_SAIDA_MODELO]
                        df_predito.columns = [f"{col}_pred" for col in df_predito.columns]
                        results_df = pd.concat([real_output_df, df_predito.set_index(real_output_df.index)], axis=1)
                        num_outputs = len(FEATURES_SAIDA_MODELO)
                        cols = 3
                        rows = int(np.ceil(num_outputs / cols))
                        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5))
                        axes = axes.flatten()
                        for i, name in enumerate(FEATURES_SAIDA_MODELO):
                            ax = axes[i]
                            real, pred = results_df[name], results_df[f"{name}_pred"]
                            r2, rmse = r2_score(real, pred), np.sqrt(mean_squared_error(real, pred))
                            err_pct = np.abs((real - pred) / real.replace(0, np.nan)) * 100
                            ax.scatter(real, pred, alpha=0.6, edgecolors="k", s=50)
                            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
                            ax.plot(lims, lims, "r--", alpha=0.75, lw=2, label="Correlação Perfeita")
                            ax.set_title(name, fontsize=14, weight="bold")
                            metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.3f}\nErro Médio: {err_pct.mean():.2f}%\nErro Máx: {err_pct.max():.2f}%"
                            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, va="top", bbox=props)
                            ax.set_xlabel(f"Valor Real ({name})")
                            ax.set_ylabel(f"Valor Previsto ({name}_pred)")
                            ax.grid(True, linestyle="--", alpha=0.5)
                            ax.legend()
                        for j in range(num_outputs, len(axes)):
                            fig.delaxes(axes[j])
                        fig.suptitle("Correlação entre Dados Reais e Previstos", fontsize=20, weight="bold")
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        st.pyplot(fig)
                        st.download_button(
                            "Download Resultados Completos (.csv)",
                            convert_df_to_csv(results_df),
                            "correlation_results.csv",
                            "text/csv",
                        )
            del st.session_state.real_df_to_process
