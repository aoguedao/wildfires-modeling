BUILDING_COLUMNS = [
    "id",
    "año",
    "material",
    "npisos",
    "h_viv",
    "n_daño",
    "orientacio",
    "elevación",
    "pendiente",
    "dist_veget",
    "dist_foco",
    "dist_estru",
    "tamaño",
    "codigo_pol",
    "sup_viv_ha",
    "raz_ocup",
    "prep_vivie",
    "mant_viv",
    "acceso_equ",
    "ac_supresi",
    "fact_agua",
    "n_conjman",
    "d_conj_man",
    "n_conj10m",
    "d_conj_10m",
    "n_conj20m",
    "d_conj_20m",
    "cord_x",
    "cord_y",
    "geometry"
]

INPUT_COLUMNS = ["wildfire"] + BUILDING_COLUMNS

FIX_BUILDING_COLUMN_NAMES = {
    "dist_foc_1": "dist_foco",
    "elevacion": "elevación",
    "preparacio": "prep_vivie",
    "raz_ocupac": "raz_ocup",
    "sup_ha": "sup_viv_ha",
    "tamañano_": "tamaño",
    "tamaño_vi": "tamaño"
}

SPANISH_NAMES = {
    "wildfire": "Incendio",
    "id": "ID",
    "año": "Año",
    "material": "Material",
    "npisos": "Nro. Pisos",
    "h_viv": "Altura Edificación",
    "n_daño": "Nivel de Daño Edificación",
    "orientacio": "Orientación",
    "elevación": "Elevación",
    "pendiente": "Pendiente",
    "dist_veget": "Dist. a Grupo Vegetal",
    "dist_foco": "Dist. al Foco",
    "dist_estru": "Dist. a otra Estructura",
    "tamaño": "Tamaño",
    "codigo_pol": "Nro. Polígono Identificatorio",
    "sup_viv_ha": "Superficie Vivienda",
    "raz_ocup": "Razón de Ocupación",
    "prep_vivie": "Preparación Vivienda",
    "mant_viv": "Mantención Vivienda",
    "fact_agua": "Suministro Agua Potable",
    "acceso_equ": "Acceso Equipo Emergencia",
    "ac_supresi": "Acceso Equipos de Supresión",
    "n_conjman": "Nro. Conjunto Manual",
    "d_conj_man": "Distancia al Borde del Conjunto Manual",
    "n_conj10m": "Nro. Conjunto Buffer 10m",
    "d_conj_10m": "Distancia al Borde del Conjunto Buffer 10m",
    "n_conj20m": "Nro. Conjunto Buffer 20m",
    "d_conj_20m": "Distancia al Borde del Conjunto Buffer 20m",
    "cord_x": "Coord. X",
    "cord_y": "Coord. Y",
}

NUM_COLUMNS = [
    "año",
    "npisos",
    "h_viv",
    "elevación",
    "pendiente",
    "dist_veget",
    "dist_foco",
    "dist_estru",
    "tamaño",
    # "codigo_pol",  # CATEGORICAL
    "sup_viv_ha",
    "raz_ocup",
    # "n_conjman",  # CATEGORICAL
    "d_conj_man",
    # "n_conj10m",  # CATEGORICAL
    "d_conj_10m",
    # "n_conj20m",  # CATEGORICAL
    "d_conj_20m",
    "cord_x",
    "cord_y",
]

CAT_COLUMNS = [
    "wildfire",
    "material",
    "orientacio",
    "codigo_pol",  # NUMERIC
    "prep_vivie",
    "mant_viv",
    "acceso_equ",
    "ac_supresi",
    "fact_agua",
    "n_conjman",  # NUMERIC
    "n_conj10m",  # NUMERIC
    "n_conj20m",  # NUMERIC
]

TARGET_COLUMN = "n_daño"

BINARY_TARGET_VALUES = {
    "Total": "Dañada",
    "Parcial": "Dañada",
    "Ninguno": "Sin Daño"
}

X_COLUMNS = [
    "material",
    "npisos",
    "h_viv",
    "orientacio",
    "elevación",
    "pendiente",
    "dist_veget",
    "dist_foco",
    "dist_estru",
    "tamaño",
    "sup_viv_ha",
    "raz_ocup",
    "prep_vivie",
    "mant_viv",
    "acceso_equ",
    "ac_supresi",
    "fact_agua",
    "d_conj_man",
    "d_conj_10m",
    "d_conj_20m",
    "cord_x",
    "cord_y"    
]