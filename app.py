import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Programación Lineal", layout="wide")

# --- VISTA 1: Configuración del Modelo ---
if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.title("Programación Lineal")
    st.header("Comenzamos configurando nuestro modelo")

    metodo = st.radio("Método a utilizar", ["Gráfico (2 variables)", "Simplex"])
    tipo_opt = st.radio("Tipo de optimización", ["Maximizar", "Minimizar"])

    st.subheader("Variables")
    x0_desc = st.text_input("Descripción de X0", "Pantalones (u/día)")
    x1_desc = st.text_input("Descripción de X1", "Camisas (u/día)")

    st.subheader("Restricciones")
    n_restr = st.number_input("Cantidad de restricciones", min_value=1, max_value=5, value=2, step=1)
    restricciones = []
    for i in range(n_restr):
        st.markdown(f"**Restricción R{i+1}**")
        desc = st.text_input(f"Descripción R{i+1}", key=f"desc_{i}")
        restricciones.append(desc)

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
    st.title("Cargamos los datos de nuestro modelo")
    st.write(f"**Variables:** X0: {st.session_state.x0_desc}, X1: {st.session_state.x1_desc}")
    st.write("**Restricciones:**")
    for i, desc in enumerate(st.session_state.restricciones_desc):
        st.write(f"R{i+1}: {desc}")

    st.subheader("Función objetivo")
    c0 = st.number_input("Coeficiente de X0", value=1.0)
    c1 = st.number_input("Coeficiente de X1", value=1.0)

    st.subheader("Restricciones")
    restr_data = []
    for i in range(st.session_state.n_restr):
        st.markdown(f"**R{i+1}: {st.session_state.restricciones_desc[i]}**")
        a = st.number_input(f"Coeficiente X0 (R{i+1})", key=f"a_{i}")
        b = st.number_input(f"Coeficiente X1 (R{i+1})", key=f"b_{i}")
        op = st.selectbox("Operador", ["<=", ">=", "="], key=f"op_{i}")
        rhs = st.number_input("Valor derecho", key=f"rhs_{i}")
        restr_data.append((a, b, op, rhs))

    if st.button("Siguiente"):
        st.session_state.c0 = c0
        st.session_state.c1 = c1
        st.session_state.restr_data = restr_data
        st.session_state.step = 3
    st.button("Volver", on_click=lambda: st.session_state.update({"step": 1}))
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
        c = [-st.session_state.c0, -st.session_state.c1]
    else:
        c = [st.session_state.c0, st.session_state.c1]

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
    if st.session_state.metodo == "Gráfico (2 variables)":
        st.subheader("Gráfico de la región factible (solo 2 variables)")
        fig, ax = plt.subplots()
        x = np.linspace(0, max(20, res.x[0]*2 if res.success else 20), 400)
        for i, (a, b, op, rhs) in enumerate(st.session_state.restr_data):
            if b != 0:
                y = (rhs - a * x) / b
                ax.plot(x, y, label=f"R{i+1}: {a}x0 + {b}x1 {op} {rhs}")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        if res.success:
            ax.plot(res.x[0], res.x[1], 'ro', label="Óptimo")
        st.pyplot(fig)

    st.button("Volver", on_click=lambda: st.session_state.update({"step": 2}))
