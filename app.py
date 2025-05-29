import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Programación Lineal", layout="wide")

# --- VISTA 1: Configuración del Modelo ---
if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.markdown("""
        <style>
        .big-title {font-size:2.2em; font-weight:bold; text-align:center;}
        .subtitle {font-size:1.3em; text-align:center;}
        .stButton>button {width: 100%;}
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 18px 10px 10px 10px;
            margin-bottom: 18px;
            background: #fafbfc;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-title">Comenzamos configurando nuestro modelo</div>', unsafe_allow_html=True)
    st.write("")

    # Fila: Método a utilizar y Tipo de optimización (cada uno en su card)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card" style="text-align:center;">'
                    '<div style="margin-bottom:10px;">Método a utilizar</div>', unsafe_allow_html=True)
        metodo = st.radio("", ["Gráfico", "Simplex"], horizontal=True, key="metodo_radio")
    with col2:
        st.markdown('<div class="card" style="text-align:center;">'
                    '<div style="margin-bottom:10px;">Tipo de optimización</div>', unsafe_allow_html=True)
        tipo_opt = st.radio("", ["Maximizar", "Minimizar"], horizontal=True, key="tipo_radio")

    # Variables
    st.markdown('<div class="subtitle">Variables</div>', unsafe_allow_html=True)
    var_cols = st.columns(2)
    x0_desc = var_cols[0].text_input("X0", "Pantalones (u/día)")
    x1_desc = var_cols[1].text_input("X1", "Camisas (u/día)")

    # Restricciones
    st.markdown('<div class="subtitle">Restricciones</div>', unsafe_allow_html=True)
    n_restr = st.number_input("Cantidad de restricciones", min_value=1, max_value=5, value=2, step=1)
    restr_cols = st.columns(1)
    restricciones = []
    for i in range(n_restr):
        restricciones.append(restr_cols[0].text_input(f"R{i}: Descripción de la restricción", key=f"desc_{i}"))

    st.write("")
    if st.button("Siguiente"):
        st.session_state.metodo = metodo
        st.session_state.tipo_opt = tipo_opt
        st.session_state.x0_desc = x0_desc
        st.session_state.x1_desc = x1_desc
        st.session_state.n_restr = n_restr
        st.session_state.restricciones_desc = restricciones
        st.session_state.step = 2
    st.stop()

# --- VISTA 2: Detalles del Modelo y Datos ---
if st.session_state.step == 2:
    st.markdown("""
        <style>
        .centered-container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            width: 100%;
        }
        .ref-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 18px 20px 10px 20px;
            margin-bottom: 18px;
            background: #fafbfc;
            width: 700px;
        }
        .section-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 18px 20px 10px 20px;
            margin-bottom: 18px;
            background: #fff;
            width: 700px;
        }
        .ref-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
        }
        .ref-label {
            font-size: 0.98em;
            color: #333;
            margin-bottom: 2px;
        }
        .stButton>button {width: 100%;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-title">Cargamos los datos de nuestro modelo</div>', unsafe_allow_html=True)
    st.write("")

    # Referencias
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<div class="ref-card">', unsafe_allow_html=True)
    st.markdown('<div class="ref-title">Referencias</div>', unsafe_allow_html=True)
    st.markdown('<div class="ref-label"><b>Variables:</b></div>', unsafe_allow_html=True)
    for i, desc in enumerate([st.session_state.x0_desc, st.session_state.x1_desc]):
        st.markdown(f'<div class="ref-label">X{i}: {desc}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ref-label"><b>Restricciones:</b></div>', unsafe_allow_html=True)
    for i, desc in enumerate(st.session_state.restricciones_desc):
        st.markdown(f'<div class="ref-label">R{i}: {desc}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Función objetivo y restricciones en una sola columna centrada
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="ref-title">Función objetivo</div>', unsafe_allow_html=True)
        cols_obj = st.columns([2, 1, 2, 1, 2, 1, 2])
        c0 = cols_obj[0].number_input("Coeficiente X0", value=1.0, key="c0", label_visibility="collapsed")
        cols_obj[1].markdown('<div style="text-align:center; margin-top:10px;">X0 +</div>', unsafe_allow_html=True)
        c1 = cols_obj[2].number_input("Coeficiente X1", value=1.0, key="c1", label_visibility="collapsed")
        cols_obj[3].markdown('<div style="text-align:center; margin-top:10px;">X1</div>', unsafe_allow_html=True)
        cols_obj[4].markdown('<div style="text-align:center; margin-top:10px;">=></div>', unsafe_allow_html=True)
        tipo = st.session_state.tipo_opt
        cols_obj[5].markdown(f'<div style="text-align:center; margin-top:10px;">{"MIN" if tipo=="Minimizar" else "MAX"}</div>', unsafe_allow_html=True)

        st.markdown('<div class="ref-title" style="margin-top:18px;">Restricciones</div>', unsafe_allow_html=True)
        restr_data = []
        for i in range(st.session_state.n_restr):
            cols = st.columns([2, 1, 2, 1, 2, 1, 2])
            a = cols[0].number_input(f"Coeficiente X0 (R{i})", key=f"a_{i}", label_visibility="collapsed")
            cols[1].markdown('<div style="text-align:center; margin-top:10px;">X0 +</div>', unsafe_allow_html=True)
            b = cols[2].number_input(f"Coeficiente X1 (R{i})", key=f"b_{i}", label_visibility="collapsed")
            cols[3].markdown('<div style="text-align:center; margin-top:10px;">X1</div>', unsafe_allow_html=True)
            op = cols[4].selectbox("Operador", ["<=", ">=", "="], key=f"op_{i}", label_visibility="collapsed")
            rhs = cols[5].number_input("Valor derecho", key=f"rhs_{i}", label_visibility="collapsed")
            restr_data.append((a, b, op, rhs))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        st.button("Volver", on_click=lambda: st.session_state.update({"step": 1}))
    with col_btn2:
        if st.button("Siguiente"):
            st.session_state.restr_data = restr_data
            st.session_state.step = 3
    st.stop()

# --- VISTA 3: Resultados y Gráfica ---
if st.session_state.step == 3:
    st.title("Resultados de la Programación Lineal")

    # Construir matrices para linprog
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for a, b, op, rhs in st.session_state.restr_data:
        if op == "<=":
            A_ub.append([a, b])
            b_ub.append(rhs)
        elif op == ">=":
            A_ub.append([-a, -b])
            b_ub.append(-rhs)
        else:
            A_eq.append([a, b])
            b_eq.append(rhs)

    if st.session_state.tipo_opt == "Maximizar":
        c = [-st.session_state["c0"], -st.session_state["c1"]]
    else:
        c = [st.session_state["c0"], st.session_state["c1"]]

    res = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        method="highs"
    )

    if res.success:
        resultado = -res.fun if st.session_state.tipo_opt == "Maximizar" else res.fun
        st.success(f"El resultado óptimo de la función objetivo es: {resultado:.2f}")
        st.write(f"X0 = {res.x[0]:.2f} ({st.session_state.x0_desc})")
        st.write(f"X1 = {res.x[1]:.2f} ({st.session_state.x1_desc})")
    else:
        st.error("No se encontró solución óptima.")

    # Gráfica solo si método gráfico
    if st.session_state.metodo == "Gráfico":
        st.subheader("Gráfico de la región factible (solo 2 variables)")
        fig, ax = plt.subplots()
        restr = st.session_state.restr_data

        # Definir límites de la gráfica
        x_bounds = [0, 20]
        y_bounds = [0, 20]

        # Generar todas las intersecciones posibles
        puntos = []

        # Intersección con los ejes
        for i, (a, b, op, rhs) in enumerate(restr):
            if a != 0:
                x0 = rhs / a
                if 0 <= x0 <= x_bounds[1]:
                    puntos.append([x0, 0])
            if b != 0:
                y0 = rhs / b
                if 0 <= y0 <= y_bounds[1]:
                    puntos.append([0, y0])

        # Intersección entre restricciones
        for i in range(len(restr)):
            for j in range(i+1, len(restr)):
                a1, b1, _, rhs1 = restr[i]
                a2, b2, _, rhs2 = restr[j]
                A = np.array([[a1, b1], [a2, b2]])
                if np.linalg.det(A) != 0:
                    sol = np.linalg.solve(A, np.array([rhs1, rhs2]))
                    if all(0 <= sol[k] <= x_bounds[1] for k in range(2)):
                        puntos.append(sol)

        # Filtrar puntos que cumplen TODAS las restricciones
        factibles = []
        for p in puntos:
            cumple = True
            for a, b, op, rhs in restr:
                val = a*p[0] + b*p[1]
                if op == "<=" and val > rhs + 1e-6:
                    cumple = False
                if op == ">=" and val < rhs - 1e-6:
                    cumple = False
                if op == "=" and abs(val - rhs) > 1e-6:
                    cumple = False
            if cumple:
                factibles.append(p)

        # Ordenar los puntos factibles para formar el polígono
        if len(factibles) > 2:
            factibles = np.array(factibles)
            # Ordenar por ángulo polar respecto al centroide
            centroid = np.mean(factibles, axis=0)
            angles = np.arctan2(factibles[:,1] - centroid[1], factibles[:,0] - centroid[0])
            orden = np.argsort(angles)
            factibles = factibles[orden]
            ax.fill(factibles[:,0], factibles[:,1], color='lightblue', alpha=0.4, label='Región factible')

        # Graficar restricciones
        x = np.linspace(0, x_bounds[1], 400)
        colores = ['r', 'g', 'b', 'm', 'c']
        for i, (a, b, op, rhs) in enumerate(restr):
            if b != 0:
                y = (rhs - a * x) / b
                ax.plot(x, y, colores[i%len(colores)], label=f'R{i+1}')
            else:
                if a != 0:
                    xval = rhs / a
                    ax.axvline(xval, color=colores[i%len(colores)], label=f'R{i+1}')

            # Punto óptimo
            if res.success:
                ax.plot(res.x[0], res.x[1], 'ko', label='Óptimo')
                ax.annotate(f"({res.x[0]:.2f},{res.x[1]:.2f})", (res.x[0], res.x[1]), textcoords="offset points", xytext=(10,10))

        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.button("Volver", on_click=lambda: st.session_state.update({"step": 2}))
