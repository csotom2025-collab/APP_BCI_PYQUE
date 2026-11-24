import re
from collections import defaultdict

def analizar_reporte_mrmr_ponderado(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    # 1. Extraer nodos y sus variables con su relevancia MI: {nodo: {variable: mi}}
    nodos_vars = defaultdict(dict)
    nodo_actual = None
    en_variables = False

    for linea in lineas:
        linea_strip = linea.strip()
        
        if linea_strip.startswith("Canales con mayor presencia"):
            break
            
        # Detectar nombre del nodo
        if linea_strip and not linea.startswith(" ") and not linea_strip.startswith(("Reporte", "===", "---", "Los nombres", "La inferencia", "'feature_columns'", "Modelos finales")):
            nodo_actual = linea_strip
            en_variables = False
            continue
            
        if linea_strip.startswith("Variables seleccionadas"):
            en_variables = True
            continue
            
        if en_variables and linea_strip.startswith("- "):
            match = re.match(r"-\s+([^\s(]+)\s+\(relevancia MI=([0-9.]+)\)", linea_strip)
            if match and nodo_actual:
                var_nombre = match.group(1)
                var_mi = float(match.group(2))
                if var_nombre not in nodos_vars[nodo_actual] or var_mi > nodos_vars[nodo_actual][var_nombre]:
                    nodos_vars[nodo_actual][var_nombre] = var_mi

    # 2. Configuración de llaves y grupos
    w_stats_full = [
        "energy", "rel", "std", "mean", "var", "rms", "skewness", "kurtosis",
        "total_energy", "energy_ratio", "mean_abs", "max", "min", "median", "iqr",
        "entropy", "zero_crossings"
    ]
    w_stats_est = ["std", "mean", "var", "rms", "skewness", "kurtosis"]

    wav_keys = [f"wA4_{s}" for s in w_stats_full] + \
               [f"wD{lvl}_{s}" for lvl in range(1, 5) for s in w_stats_full]

    wAD_Est = [f"wA4_{s}" for s in w_stats_est] + \
              [f"wD{lvl}_{s}" for lvl in range(1, 5) for s in w_stats_est]

    bandas_fft = ["_delta_", "_theta_", "_alpha_", "_beta_", "_gamma_"]

    def clasificar_sets(valid_cols):
        return {
            "Estadisticas_Tm": [c for c in valid_cols if "_Tm_" in c],
            "Frecuencias_Abs": [c for c in valid_cols if any(k in c for k in ["delta_Abs", "theta_Abs", "alpha_Abs", "beta_Abs", "gamma_Abs"])],
            "Frecuencias_Rel": [c for c in valid_cols if any(k in c for k in ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"])],
            "Frecuencias_Est": [c for c in valid_cols if any(k in c for k in [
                "delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                "beta_std", "gamma_std", "beta_var", "gamma_var", "beta_rms", "gamma_rms",
                "beta_skewness", "gamma_skewness", "beta_kurtosis", "gamma_kurtosis"
            ])],
            "Wavelets": [c for c in valid_cols if any(k in c for k in wav_keys)],
            "Wavelets_Est": [c for c in valid_cols if any(k in c for k in wAD_Est)],
            "Frecuencias_Todas": [c for c in valid_cols if any(b in c for b in bandas_fft) and "_welch_" not in c],
            "Welch": [c for c in valid_cols if "_welch_" in c],
            "Estadicas_todas": [c for c in valid_cols if any(c.endswith(s) for s in ["_mean", "_std", "_var", "_rms", "_skewness", "_kurtosis"])],
            "Beta_Gamma_All": [c for c in valid_cols if any(b in c for b in ["_beta_", "_gamma_"]) and "_welch_" not in c],
            "Beta_Gamma_Est": [c for c in valid_cols if any(b in c for b in ["_beta_", "_gamma_"]) and "_welch_" not in c and not c.endswith(("_Abs", "_rel"))],
            "ERP_P300_Puro": [c for c in valid_cols if "_p300_" in c],
            "Bajas_Frec_ERP": [c for c in valid_cols if any(k in c for k in ["_delta_", "_theta_", "_alpha_", "_wA4_", "_wD4_", "_p300_"])],
            "Parieto_Occipital_All": [c for c in valid_cols if c.startswith(("P7_", "P8_", "O1_", "O2_"))],
            "Beta_Gamma_ERP": [c for c in valid_cols if (any(b in c for b in ["_beta_", "_gamma_"]) and "_welch_" not in c) or "_p300_" in c],
            "Espectral_Mas_ERP": [c for c in valid_cols if any(k in c for k in ["_welch_", "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel", "_p300_"])],
            "Relativas_Mas_ERP": [c for c in valid_cols if any(k in c for k in ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel", "_p300_"])],
        }

    # Acumuladores para el resumen general global
    global_canales = defaultdict(lambda: {"mi": 0.0, "count": 0, "nodos": set()})
    global_variables = defaultdict(lambda: {"mi": 0.0, "count": 0, "nodos": set()})
    global_calculos = defaultdict(lambda: {"mi": 0.0, "count": 0, "nodos": set()})
    global_sets = defaultdict(lambda: {"mi": 0.0, "count": 0, "nodos": set()})
    total_mi_global = 0.0
    total_vars_global = 0
    nodos_activos = 0

    # 3. Procesar e imprimir resultados ponderados por nodo
    for nodo, dict_vars in nodos_vars.items():
        if not dict_vars:
            continue
            
        valid_dict = {c: mi for c, mi in dict_vars.items() if "_wD5_" not in c and "num_coeffs" not in c}
        valid_cols = list(valid_dict.keys())
        total_vars = len(valid_cols)
        total_mi = sum(valid_dict.values())

        nodos_activos += 1
        total_vars_global += total_vars
        total_mi_global += total_mi
        
        print("=" * 80)
        print(f"NODO / NIVEL: {nodo} | Vars válidas: {total_vars} | MI Acumulado Total: {total_mi:.4f}")
        print("=" * 80)

        # --- A. Canales ponderados por MI ---
        mi_canales = defaultdict(lambda: {"mi": 0.0, "count": 0})
        for col, mi in valid_dict.items():
            canal = col.split("_")[0]
            mi_canales[canal]["mi"] += mi
            mi_canales[canal]["count"] += 1
            
            # Acumular en global
            global_canales[canal]["mi"] += mi
            global_canales[canal]["count"] += 1
            global_canales[canal]["nodos"].add(nodo)

            global_variables[col]["mi"] += mi
            global_variables[col]["count"] += 1
            global_variables[col]["nodos"].add(nodo)

        canales_ordenados = sorted(mi_canales.items(), key=lambda x: x[1]["mi"], reverse=True)
        print("\n Canales ordenados por Relevancia MI acumulada:")
        for canal, datos in canales_ordenados:
            pct_mi = (datos["mi"] / total_mi * 100) if total_mi > 0 else 0
            media_mi = datos["mi"] / datos["count"]
            print(f"  - {canal:<4}: MI Suma = {datos['mi']:>7.4f} ({pct_mi:>5.1f}%) | Vars = {datos['count']:>3} | MI Media = {media_mi:.4f}")

        # --- B. Cálculos específicos ponderados por MI ---
        mi_calculos = defaultdict(lambda: {"mi": 0.0, "count": 0})
        for col, mi in valid_dict.items():
            if "_" in col:
                calc = col.split("_", 1)[1]
                mi_calculos[calc]["mi"] += mi
                mi_calculos[calc]["count"] += 1

                # Acumular en global
                global_calculos[calc]["mi"] += mi
                global_calculos[calc]["count"] += 1
                global_calculos[calc]["nodos"].add(nodo)

        calculos_ordenados = sorted(mi_calculos.items(), key=lambda x: x[1]["mi"], reverse=True)[:10]
        print("\n Top 10 cálculos específicos por Relevancia MI acumulada:")
        for calc, datos in calculos_ordenados:
            pct_mi = (datos["mi"] / total_mi * 100) if total_mi > 0 else 0
            media_mi = datos["mi"] / datos["count"]
            print(f"  - {calc:<25}: MI Suma = {datos['mi']:>7.4f} ({pct_mi:>5.1f}%) | Vars = {datos['count']:>3} | MI Media = {media_mi:.4f}")

        # --- C. Agrupación por sets ponderada por MI ---
        sets_resultado = clasificar_sets(valid_cols)
        sets_ponderados = []
        for nombre_set, lista_vars in sets_resultado.items():
            if lista_vars:
                suma_mi_set = sum(valid_dict[c] for c in lista_vars)
                sets_ponderados.append((nombre_set, suma_mi_set, len(lista_vars)))

                # Acumular en global
                global_sets[nombre_set]["mi"] += suma_mi_set
                global_sets[nombre_set]["count"] += len(lista_vars)
                global_sets[nombre_set]["nodos"].add(nodo)

        sets_ponderados.sort(key=lambda x: x[1], reverse=True)
        print("\n Grupos (sets) ordenados por Relevancia MI acumulada:")
        for nombre_set, suma_mi, count in sets_ponderados:
            pct_mi = (suma_mi / total_mi * 100) if total_mi > 0 else 0
            media_mi = suma_mi / count
            print(f"  - {nombre_set:<22}: MI Suma = {suma_mi:>7.4f} ({pct_mi:>5.1f}%) | Vars = {count:>3} | MI Media = {media_mi:.4f}")
        print("\n")

    # ============================================================================
    # 4. RESUMEN GENERAL GLOBAL (CONSOLIDADO DE TODOS LOS NODOS)
    # ============================================================================
    print("#" * 80)
    print(f"RESUMEN GENERAL GLOBAL | Nodos analizados: {nodos_activos} | Total Instancias Vars: {total_vars_global} | MI Global: {total_mi_global:.4f}")
    print("#" * 80)

    # 4.1 Canales Globales
    g_canales_ord = sorted(global_canales.items(), key=lambda x: x[1]["mi"], reverse=True)
    print("\n [GLOBAL] Canales con mayor Relevancia MI acumulada:")
    for canal, datos in g_canales_ord:
        pct = (datos["mi"] / total_mi_global * 100) if total_mi_global > 0 else 0
        media = datos["mi"] / datos["count"]
        print(f"  - {canal:<4}: MI Suma = {datos['mi']:>8.4f} ({pct:>5.1f}%) | Apariciones = {datos['count']:>4} | Nodos = {len(datos['nodos'])}/{nodos_activos} | MI Media = {media:.4f}")

    # 4.2 Variables Exactas Globales (Top 15)
    g_vars_ord = sorted(global_variables.items(), key=lambda x: x[1]["mi"], reverse=True)[:15]
    print("\n [GLOBAL] Top 15 Variables exactas (Canal + Cálculo) con mayor Relevancia MI:")
    for var, datos in g_vars_ord:
        pct = (datos["mi"] / total_mi_global * 100) if total_mi_global > 0 else 0
        media = datos["mi"] / datos["count"]
    print(f"  - {var:<28}: MI Suma = {datos['mi']:>8.4f} ({pct:>5.1f}%) | Apariciones = {datos['count']:>2} | Nodos = {len(datos['nodos'])}/{nodos_activos} | MI Media = {media:.4f}")
    # 4.3 Tipos de Cálculo Globales (Top 15)
    g_calc_ord = sorted(global_calculos.items(), key=lambda x: x[1]["mi"], reverse=True)[:15]
    print("\n [GLOBAL] Top 15 Cálculos (sin canal) con mayor Relevancia MI:")
    for calc, datos in g_calc_ord:
        pct = (datos["mi"] / total_mi_global * 100) if total_mi_global > 0 else 0
        media = datos["mi"] / datos["count"]
    print(f"  - {calc:<25}: MI Suma = {datos['mi']:>8.4f} ({pct:>5.1f}%) | Apariciones = {datos['count']:>4} | Nodos = {len(datos['nodos'])}/{nodos_activos} | MI Media = {media:.4f}")
    # 4.4 Grupos (Sets) Globales
    g_sets_ord = sorted(global_sets.items(), key=lambda x: x[1]["mi"], reverse=True)
    print("\n [GLOBAL] Grupos (sets) con mayor Relevancia MI acumulada:")
    for nombre_set, datos in g_sets_ord:
        pct = (datos["mi"] / total_mi_global * 100) if total_mi_global > 0 else 0
        media = datos["mi"] / datos["count"]
        print(f"  - {nombre_set:<22}: MI Suma = {datos['mi']:>8.4f} ({pct:>5.1f}%) | Apariciones = {datos['count']:>4} | Nodos = {len(datos['nodos'])}/{nodos_activos} | MI Media = {media:.4f}")
    print("\n")
if __name__ == "__main__":
    analizar_reporte_mrmr_ponderado(r"C:/Users/usuario/Desktop/APP_BCI_PYQUE/trainingOutputs/UserCMSM/reporte_mrmr_40_rep.txt")