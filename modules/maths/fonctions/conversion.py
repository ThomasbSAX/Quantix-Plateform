"""
Module de conversion d'unités
Permet de convertir différentes unités de mesure entre elles
"""

# Facteurs de conversion vers l'unité de base (SI)
UNITES = {
    'longueur': {
        'base': 'metre',
        'unites': {
            'metre': 1,
            'm': 1,
            'kilometre': 1000,
            'km': 1000,
            'centimetre': 0.01,
            'cm': 0.01,
            'millimetre': 0.001,
            'mm': 0.001,
            'mile': 1609.34,
            'mi': 1609.34,
            'yard': 0.9144,
            'yd': 0.9144,
            'pied': 0.3048,
            'ft': 0.3048,
            'pouce': 0.0254,
            'in': 0.0254,
        }
    },
    'masse': {
        'base': 'kilogramme',
        'unites': {
            'kilogramme': 1,
            'kg': 1,
            'gramme': 0.001,
            'g': 0.001,
            'milligramme': 0.000001,
            'mg': 0.000001,
            'tonne': 1000,
            't': 1000,
            'livre': 0.453592,
            'lb': 0.453592,
            'once': 0.0283495,
            'oz': 0.0283495,
        }
    },
    'temps': {
        'base': 'seconde',
        'unites': {
            'seconde': 1,
            's': 1,
            'minute': 60,
            'min': 60,
            'heure': 3600,
            'h': 3600,
            'jour': 86400,
            'j': 86400,
            'semaine': 604800,
            'mois': 2592000,  # 30 jours
            'annee': 31536000,  # 365 jours
        }
    },
    'volume': {
        'base': 'litre',
        'unites': {
            'litre': 1,
            'l': 1,
            'millilitre': 0.001,
            'ml': 0.001,
            'centilitre': 0.01,
            'cl': 0.01,
            'decilitre': 0.1,
            'dl': 0.1,
            'hectolitre': 100,
            'hl': 100,
            'metre_cube': 1000,
            'm3': 1000,
            'gallon': 3.78541,  # gallon US
            'gal': 3.78541,
            'pinte': 0.473176,  # pinte US
            'pt': 0.473176,
        }
    },
    'surface': {
        'base': 'metre_carre',
        'unites': {
            'metre_carre': 1,
            'm2': 1,
            'kilometre_carre': 1000000,
            'km2': 1000000,
            'centimetre_carre': 0.0001,
            'cm2': 0.0001,
            'hectare': 10000,
            'ha': 10000,
            'are': 100,
            'a': 100,
            'acre': 4046.86,
        }
    },
    'vitesse': {
        'base': 'metre_par_seconde',
        'unites': {
            'metre_par_seconde': 1,
            'm/s': 1,
            'kilometre_par_heure': 0.277778,
            'km/h': 0.277778,
            'mile_par_heure': 0.44704,
            'mph': 0.44704,
            'noeud': 0.514444,
            'kt': 0.514444,
        }
    },
    'energie': {
        'base': 'joule',
        'unites': {
            'joule': 1,
            'j': 1,
            'kilojoule': 1000,
            'kj': 1000,
            'calorie': 4.184,
            'cal': 4.184,
            'kilocalorie': 4184,
            'kcal': 4184,
            'watt_heure': 3600,
            'wh': 3600,
            'kilowatt_heure': 3600000,
            'kwh': 3600000,
        }
    },
    'pression': {
        'base': 'pascal',
        'unites': {
            'pascal': 1,
            'pa': 1,
            'hectopascal': 100,
            'hpa': 100,
            'kilopascal': 1000,
            'kpa': 1000,
            'bar': 100000,
            'millibar': 100,
            'mbar': 100,
            'atmosphere': 101325,
            'atm': 101325,
            'psi': 6894.76,
            'mmhg': 133.322,  # millimètre de mercure
            'torr': 133.322,
        }
    },
    'puissance': {
        'base': 'watt',
        'unites': {
            'watt': 1,
            'w': 1,
            'kilowatt': 1000,
            'kw': 1000,
            'megawatt': 1000000,
            'mw': 1000000,
            'cheval_vapeur': 735.5,  # cheval vapeur métrique
            'cv': 735.5,
            'horsepower': 745.7,  # HP britannique
            'hp': 745.7,
        }
    },
    'force': {
        'base': 'newton',
        'unites': {
            'newton': 1,
            'n': 1,
            'kilonewton': 1000,
            'kn': 1000,
            'dyne': 0.00001,
            'kilogramme_force': 9.80665,
            'kgf': 9.80665,
            'livre_force': 4.44822,
            'lbf': 4.44822,
        }
    },
    'angle': {
        'base': 'radian',
        'unites': {
            'radian': 1,
            'rad': 1,
            'degre': 0.0174533,
            'deg': 0.0174533,
            '°': 0.0174533,
            'grade': 0.015708,
            'gon': 0.015708,
            'tour': 6.28319,  # tour complet
            'revolution': 6.28319,
        }
    },
    'donnees': {
        'base': 'octet',
        'unites': {
            'bit': 0.125,
            'octet': 1,
            'byte': 1,
            'o': 1,
            'kilooctet': 1024,
            'ko': 1024,
            'kb': 1024,
            'megaoctet': 1048576,
            'mo': 1048576,
            'mb': 1048576,
            'gigaoctet': 1073741824,
            'go': 1073741824,
            'gb': 1073741824,
            'teraoctet': 1099511627776,
            'to': 1099511627776,
            'tb': 1099511627776,
            'petaoctet': 1125899906842624,
            'po': 1125899906842624,
            'pb': 1125899906842624,
        }
    },
    'frequence': {
        'base': 'hertz',
        'unites': {
            'hertz': 1,
            'hz': 1,
            'kilohertz': 1000,
            'khz': 1000,
            'megahertz': 1000000,
            'mhz': 1000000,
            'gigahertz': 1000000000,
            'ghz': 1000000000,
        }
    },
    'tension': {
        'base': 'volt',
        'unites': {
            'volt': 1,
            'v': 1,
            'millivolt': 0.001,
            'mv': 0.001,
            'kilovolt': 1000,
            'kv': 1000,
        }
    },
    'courant': {
        'base': 'ampere',
        'unites': {
            'ampere': 1,
            'a': 1,
            'milliampere': 0.001,
            'ma': 0.001,
            'microampere': 0.000001,
            'μa': 0.000001,
            'kiloampere': 1000,
            'ka': 1000,
        }
    },
    'resistance': {
        'base': 'ohm',
        'unites': {
            'ohm': 1,
            'Ω': 1,
            'milliohm': 0.001,
            'mΩ': 0.001,
            'kiloohm': 1000,
            'kΩ': 1000,
            'megaohm': 1000000,
            'MΩ': 1000000,
        }
    },
    'densite': {
        'base': 'kg_par_m3',
        'unites': {
            'kg_par_m3': 1,
            'kg/m3': 1,
            'g_par_cm3': 1000,
            'g/cm3': 1000,
            'g_par_l': 1,
            'g/l': 1,
            'lb_par_ft3': 16.0185,
            'lb/ft3': 16.0185,
        }
    },
    'debit': {
        'base': 'litre_par_seconde',
        'unites': {
            'litre_par_seconde': 1,
            'l/s': 1,
            'litre_par_minute': 0.0166667,
            'l/min': 0.0166667,
            'litre_par_heure': 0.000277778,
            'l/h': 0.000277778,
            'metre_cube_par_heure': 0.277778,
            'm3/h': 0.277778,
            'gallon_par_minute': 0.0630902,
            'gpm': 0.0630902,
        }
    },
    'acceleration': {
        'base': 'metre_par_seconde_carree',
        'unites': {
            'metre_par_seconde_carree': 1,
            'm/s2': 1,
            'm/s²': 1,
            'g': 9.80665,  # gravité terrestre
            'gal': 0.01,  # galileo
        }
    },
    'luminosite': {
        'base': 'lumen',
        'unites': {
            'lumen': 1,
            'lm': 1,
        }
    },
    'eclairement': {
        'base': 'lux',
        'unites': {
            'lux': 1,
            'lx': 1,
            'footcandle': 10.764,
            'fc': 10.764,
        }
    },
    'radioactivite': {
        'base': 'becquerel',
        'unites': {
            'becquerel': 1,
            'bq': 1,
            'curie': 37000000000,
            'ci': 37000000000,
            'millicurie': 37000000,
            'mci': 37000000,
            'microcurie': 37000,
            'μci': 37000,
        }
    },
    'dose_absorbee': {
        'base': 'gray',
        'unites': {
            'gray': 1,
            'gy': 1,
            'rad': 0.01,
            'milligray': 0.001,
            'mgy': 0.001,
        }
    },
    'capacite': {
        'base': 'farad',
        'unites': {
            'farad': 1,
            'f': 1,
            'microfarad': 0.000001,
            'μf': 0.000001,
            'nanofarad': 0.000000001,
            'nf': 0.000000001,
            'picofarad': 0.000000000001,
            'pf': 0.000000000001,
        }
    },
    'inductance': {
        'base': 'henry',
        'unites': {
            'henry': 1,
            'h': 1,
            'millihenry': 0.001,
            'mh': 0.001,
            'microhenry': 0.000001,
            'μh': 0.000001,
        }
    },
    'concentration': {
        'base': 'mol_par_litre',
        'unites': {
            'mol_par_litre': 1,
            'mol/l': 1,
            'm': 1,  # molarité
            'millimol_par_litre': 0.001,
            'mmol/l': 0.001,
            'micromol_par_litre': 0.000001,
            'μmol/l': 0.000001,
        }
    },
    'bande_passante': {
        'base': 'bit_par_seconde',
        'unites': {
            'bit_par_seconde': 1,
            'bps': 1,
            'bit/s': 1,
            'kilobit_par_seconde': 1000,
            'kbps': 1000,
            'kb/s': 1000,
            'megabit_par_seconde': 1000000,
            'mbps': 1000000,
            'mb/s': 1000000,
            'gigabit_par_seconde': 1000000000,
            'gbps': 1000000000,
            'gb/s': 1000000000,
            'octet_par_seconde': 8,
            'o/s': 8,
            'kilooctet_par_seconde': 8192,
            'ko/s': 8192,
            'megaoctet_par_seconde': 8388608,
            'mo/s': 8388608,
            'gigaoctet_par_seconde': 8589934592,
            'go/s': 8589934592,
        }
    },
    'resolution': {
        'base': 'pixel',
        'unites': {
            'pixel': 1,
            'px': 1,
            'megapixel': 1000000,
            'mp': 1000000,
            'sd_480p': 307200,  # 640x480
            'hd_720p': 921600,  # 1280x720
            'full_hd_1080p': 2073600,  # 1920x1080
            '2k': 2073600,
            '4k_uhd': 8294400,  # 3840x2160
            '4k': 8294400,
            '8k_uhd': 33177600,  # 7680x4320
            '8k': 33177600,
        }
    },
    'dpi': {
        'base': 'dpi',
        'unites': {
            'dpi': 1,  # dots per inch
            'ppi': 1,  # pixels per inch
            'point_par_pouce': 1,
            'dpcm': 2.54,  # dots per centimeter
            'point_par_cm': 2.54,
        }
    },
    'taux_rafraichissement': {
        'base': 'fps',
        'unites': {
            'fps': 1,  # frames per second
            'image_par_seconde': 1,
            'ips': 1,
            'hertz_video': 1,  # équivalent pour affichage
        }
    },
    'latence': {
        'base': 'milliseconde',
        'unites': {
            'milliseconde': 1,
            'ms': 1,
            'microseconde': 0.001,
            'μs': 0.001,
            'nanoseconde': 0.000001,
            'ns': 0.000001,
            'seconde': 1000,
        }
    },
    'debit_binaire': {
        'base': 'kbps',
        'unites': {
            'kbps': 1,  # kilobit par seconde
            'kb/s': 1,
            'mbps': 1000,
            'mb/s': 1000,
            'mp3_128': 128,  # débit MP3 standard
            'mp3_320': 320,  # débit MP3 haute qualité
            'aac_256': 256,  # débit AAC
            'cd_audio': 1411.2,  # 44.1kHz 16bit stéréo
            'flac': 1411.2,
        }
    },
    'temps_cpu': {
        'base': 'cycle',
        'unites': {
            'cycle': 1,
            'instruction': 1,  # approximation
            'nanoseconde_cpu': 1,  # pour 1GHz
        }
    },
    'adressage_memoire': {
        'base': 'adresse',
        'unites': {
            'adresse': 1,
            'bit_adresse': 1,
            '8_bit': 256,  # 2^8 adresses
            '16_bit': 65536,  # 2^16 adresses
            '32_bit': 4294967296,  # 2^32 adresses
            '64_bit': 18446744073709551616,  # 2^64 adresses
        }
    },
    'taille_police': {
        'base': 'point',
        'unites': {
            'point': 1,  # point typographique
            'pt': 1,
            'pixel_72dpi': 1,  # à 72 DPI, 1pt = 1px
            'pixel_96dpi': 1.333,  # à 96 DPI
            'pica': 12,  # 1 pica = 12 points
            'em': 1,  # relatif à la taille courante
        }
    },
    'couleur': {
        'base': 'bit',
        'unites': {
            'bit': 1,
            'bit_par_pixel': 1,
            'bpp': 1,
            '8_bit': 8,  # 256 couleurs
            '16_bit': 16,  # 65K couleurs
            '24_bit': 24,  # 16.7M couleurs (True Color)
            '32_bit': 32,  # 16.7M + alpha
            '48_bit': 48,  # HDR
        }
    },
    'compression': {
        'base': 'ratio',
        'unites': {
            'ratio': 1,
            'pourcentage': 0.01,
            '%': 0.01,
            '2_pour_1': 2,
            '5_pour_1': 5,
            '10_pour_1': 10,
            'zip': 2.5,  # compression moyenne
            'gzip': 2.5,
            'jpeg': 10,  # compression moyenne
            'png': 1.5,  # sans perte
        }
    },
    'duree_video': {
        'base': 'seconde_video',
        'unites': {
            'seconde_video': 1,
            'frame_24fps': 0.0416667,  # 1/24
            'frame_25fps': 0.04,  # 1/25 (PAL)
            'frame_30fps': 0.0333333,  # 1/30 (NTSC)
            'frame_60fps': 0.0166667,  # 1/60
            'minute_video': 60,
            'heure_video': 3600,
        }
    },
    'iops': {
        'base': 'operations_par_seconde',
        'unites': {
            'operations_par_seconde': 1,
            'ops': 1,
            'iops': 1,  # Input/Output Operations Per Second
            'kiops': 1000,
            'miops': 1000000,
        }
    },
    'throughput': {
        'base': 'transaction_par_seconde',
        'unites': {
            'transaction_par_seconde': 1,
            'tps': 1,
            'requete_par_seconde': 1,
            'rps': 1,
            'qps': 1,  # queries per second
            'transaction_par_minute': 0.0166667,
            'tpm': 0.0166667,
        }
    },
    'version_logiciel': {
        'base': 'version',
        'unites': {
            'version': 1,
            'majeure': 1,
            'mineure': 0.1,
            'patch': 0.01,
            'build': 0.0001,
        }
    },
    'charge_electrique': {
        'base': 'coulomb',
        'unites': {
            'coulomb': 1,
            'c': 1,
            'millicoulomb': 0.001,
            'mc': 0.001,
            'microcoulomb': 0.000001,
            'μc': 0.000001,
            'ampere_heure': 3600,
            'ah': 3600,
            'milliampere_heure': 3.6,
            'mah': 3.6,
            'electron': 1.602176634e-19,  # charge d'un électron
        }
    },
    'champ_electrique': {
        'base': 'volt_par_metre',
        'unites': {
            'volt_par_metre': 1,
            'v/m': 1,
            'kilovolt_par_metre': 1000,
            'kv/m': 1000,
            'volt_par_centimetre': 100,
            'v/cm': 100,
            'newton_par_coulomb': 1,
            'n/c': 1,
        }
    },
    'champ_magnetique': {
        'base': 'tesla',
        'unites': {
            'tesla': 1,
            't': 1,
            'millitesla': 0.001,
            'mt': 0.001,
            'microtesla': 0.000001,
            'μt': 0.000001,
            'gauss': 0.0001,
            'g': 0.0001,
        }
    },
    'flux_magnetique': {
        'base': 'weber',
        'unites': {
            'weber': 1,
            'wb': 1,
            'milliweber': 0.001,
            'mwb': 0.001,
            'maxwell': 0.00000001,
            'mx': 0.00000001,
        }
    },
    'conductance': {
        'base': 'siemens',
        'unites': {
            'siemens': 1,
            's': 1,
            'millisiemens': 0.001,
            'ms': 0.001,
            'microsiemens': 0.000001,
            'μs': 0.000001,
            'mho': 1,  # ancienne unité
        }
    },
    'couple': {
        'base': 'newton_metre',
        'unites': {
            'newton_metre': 1,
            'nm': 1,
            'n·m': 1,
            'kilonewton_metre': 1000,
            'knm': 1000,
            'livre_pied': 1.35582,
            'lbf·ft': 1.35582,
            'livre_pouce': 0.112985,
            'lbf·in': 0.112985,
            'dyne_centimetre': 0.0000001,
        }
    },
    'moment_inertie': {
        'base': 'kg_m2',
        'unites': {
            'kg_m2': 1,
            'kg·m²': 1,
            'g_cm2': 0.0000001,
            'g·cm²': 0.0000001,
            'lb_ft2': 0.0421401,
            'lb·ft²': 0.0421401,
        }
    },
    'viscosite_dynamique': {
        'base': 'pascal_seconde',
        'unites': {
            'pascal_seconde': 1,
            'pa·s': 1,
            'millipascal_seconde': 0.001,
            'mpa·s': 0.001,
            'poise': 0.1,
            'p': 0.1,
            'centipoise': 0.001,
            'cp': 0.001,
        }
    },
    'viscosite_cinematique': {
        'base': 'm2_par_seconde',
        'unites': {
            'm2_par_seconde': 1,
            'm²/s': 1,
            'stokes': 0.0001,
            'st': 0.0001,
            'centistokes': 0.000001,
            'cst': 0.000001,
        }
    },
    'conductivite_thermique': {
        'base': 'watt_par_m_k',
        'unites': {
            'watt_par_m_k': 1,
            'w/(m·k)': 1,
            'w/m·k': 1,
            'cal_par_s_cm_c': 418.4,
            'btu_par_h_ft_f': 1.73073,
        }
    },
    'capacite_thermique': {
        'base': 'joule_par_kelvin',
        'unites': {
            'joule_par_kelvin': 1,
            'j/k': 1,
            'kilojoule_par_kelvin': 1000,
            'kj/k': 1000,
            'calorie_par_celsius': 4.184,
            'cal/°c': 4.184,
        }
    },
    'chaleur_specifique': {
        'base': 'joule_par_kg_k',
        'unites': {
            'joule_par_kg_k': 1,
            'j/(kg·k)': 1,
            'kilojoule_par_kg_k': 1000,
            'kj/(kg·k)': 1000,
            'calorie_par_g_c': 4184,
            'cal/(g·°c)': 4184,
            'btu_par_lb_f': 4186.8,
        }
    },
    'intensite_lumineuse': {
        'base': 'candela',
        'unites': {
            'candela': 1,
            'cd': 1,
            'millicandela': 0.001,
            'mcd': 0.001,
            'bougie': 1,  # approximation
        }
    },
    'entropie': {
        'base': 'joule_par_kelvin',
        'unites': {
            'joule_par_kelvin': 1,
            'j/k': 1,
            'kilojoule_par_kelvin': 1000,
            'kj/k': 1000,
            'calorie_par_kelvin': 4.184,
            'cal/k': 4.184,
        }
    },
    'quantite_mouvement': {
        'base': 'kg_m_par_s',
        'unites': {
            'kg_m_par_s': 1,
            'kg·m/s': 1,
            'newton_seconde': 1,
            'n·s': 1,
            'g_cm_par_s': 0.00001,
        }
    },
    'impulsion': {
        'base': 'newton_seconde',
        'unites': {
            'newton_seconde': 1,
            'n·s': 1,
            'kg_m_par_s': 1,
            'kg·m/s': 1,
            'dyne_seconde': 0.00001,
        }
    },
    'debit_massique': {
        'base': 'kg_par_s',
        'unites': {
            'kg_par_s': 1,
            'kg/s': 1,
            'g_par_s': 0.001,
            'g/s': 0.001,
            'kg_par_h': 0.000277778,
            'kg/h': 0.000277778,
            'tonne_par_h': 0.277778,
            't/h': 0.277778,
            'lb_par_s': 0.453592,
            'lb/s': 0.453592,
        }
    },
    'pression_partielle': {
        'base': 'pascal',
        'unites': {
            'pascal': 1,
            'pa': 1,
            'kilopascal': 1000,
            'kpa': 1000,
            'bar': 100000,
            'mmhg': 133.322,
            'torr': 133.322,
        }
    },
    'coefficient_transfert_thermique': {
        'base': 'watt_par_m2_k',
        'unites': {
            'watt_par_m2_k': 1,
            'w/(m²·k)': 1,
            'cal_par_s_m2_c': 4184,
            'btu_par_h_ft2_f': 5.67826,
        }
    },
    'flux_thermique': {
        'base': 'watt_par_m2',
        'unites': {
            'watt_par_m2': 1,
            'w/m²': 1,
            'kilowatt_par_m2': 1000,
            'kw/m²': 1000,
            'btu_par_h_ft2': 3.15459,
            'cal_par_s_cm2': 41840,
        }
    },
    'activite_catalytique': {
        'base': 'katal',
        'unites': {
            'katal': 1,
            'kat': 1,
            'mol_par_seconde': 1,
            'mol/s': 1,
            'unite_enzyme': 0.0000000166667,  # 1 µmol/min
            'u': 0.0000000166667,
        }
    },
    'dose_equivalente': {
        'base': 'sievert',
        'unites': {
            'sievert': 1,
            'sv': 1,
            'millisievert': 0.001,
            'msv': 0.001,
            'microsievert': 0.000001,
            'μsv': 0.000001,
            'rem': 0.01,
            'millirem': 0.00001,
            'mrem': 0.00001,
        }
    },
    'exposition_rayonnement': {
        'base': 'coulomb_par_kg',
        'unites': {
            'coulomb_par_kg': 1,
            'c/kg': 1,
            'roentgen': 0.000258,
            'r': 0.000258,
            'milliroentgen': 0.000000258,
            'mr': 0.000000258,
        }
    },
    'potentiel_electrique': {
        'base': 'volt',
        'unites': {
            'volt': 1,
            'v': 1,
            'millivolt': 0.001,
            'mv': 0.001,
            'microvolt': 0.000001,
            'μv': 0.000001,
            'kilovolt': 1000,
            'kv': 1000,
            'megavolt': 1000000,
            'megav': 1000000,
        }
    },
    'permeabilite': {
        'base': 'henry_par_metre',
        'unites': {
            'henry_par_metre': 1,
            'h/m': 1,
            'microhenry_par_metre': 0.000001,
            'μh/m': 0.000001,
            'nanohenry_par_metre': 0.000000001,
            'nh/m': 0.000000001,
        }
    },
    'permittivite': {
        'base': 'farad_par_metre',
        'unites': {
            'farad_par_metre': 1,
            'f/m': 1,
            'picofarad_par_metre': 0.000000000001,
            'pf/m': 0.000000000001,
        }
    },
    'quantite_matiere': {
        'base': 'mole',
        'unites': {
            'mole': 1,
            'mol': 1,
            'millimole': 0.001,
            'mmol': 0.001,
            'micromole': 0.000001,
            'µmol': 0.000001,
            'kilomole': 1000,
            'kmol': 1000,
            'molecule': 1.66053906660e-24,  # 1/N_A
        }
    },
    'masse_molaire': {
        'base': 'g_par_mol',
        'unites': {
            'g_par_mol': 1,
            'g/mol': 1,
            'kg_par_mol': 1000,
            'kg/mol': 1000,
            'kg_par_kmol': 1,
            'kg/kmol': 1,
        }
    },
    'ph': {
        'base': 'ph',
        'unites': {
            'ph': 1,
            'poh': 1,  # échelle inverse
        }
    },
    'constante_gaz': {
        'base': 'j_par_mol_k',
        'unites': {
            'j_par_mol_k': 1,
            'j/(mol·k)': 1,
            'cal_par_mol_k': 4.184,
            'cal/(mol·k)': 4.184,
            'l_atm_par_mol_k': 101.325,
            'l·atm/(mol·k)': 101.325,
        }
    },
    'energie_molaire': {
        'base': 'joule_par_mole',
        'unites': {
            'joule_par_mole': 1,
            'j/mol': 1,
            'kilojoule_par_mole': 1000,
            'kj/mol': 1000,
            'calorie_par_mole': 4.184,
            'cal/mol': 4.184,
            'kilocalorie_par_mole': 4184,
            'kcal/mol': 4184,
            'electronvolt': 96485.3321,  # eV par mole
            'ev/molecule': 96485.3321,
        }
    },
    'potentiel_redox': {
        'base': 'volt',
        'unites': {
            'volt': 1,
            'v': 1,
            'millivolt': 0.001,
            'mv': 0.001,
        }
    },
    'solubilite': {
        'base': 'mol_par_litre',
        'unites': {
            'mol_par_litre': 1,
            'mol/l': 1,
            'mmol_par_litre': 0.001,
            'mmol/l': 0.001,
            'g_par_litre': 1,  # approximation, dépend du composé
            'g/l': 1,
            'g_par_100ml': 10,
            'g/100ml': 10,
            'ppm': 0.001,  # parties par million (mg/L)
            'ppb': 0.000001,  # parties par milliard (µg/L)
        }
    },
    'pression_vapeur': {
        'base': 'pascal',
        'unites': {
            'pascal': 1,
            'pa': 1,
            'bar': 100000,
            'mmhg': 133.322,
            'torr': 133.322,
            'atmosphere': 101325,
            'atm': 101325,
        }
    },
    'distance_astronomique': {
        'base': 'metre',
        'unites': {
            'metre': 1,
            'm': 1,
            'kilometre': 1000,
            'km': 1000,
            'unite_astronomique': 149597870700,  # distance Terre-Soleil
            'ua': 149597870700,
            'au': 149597870700,
            'annee_lumiere': 9460730472580800,
            'al': 9460730472580800,
            'ly': 9460730472580800,
            'parsec': 30856775814913673,
            'pc': 30856775814913673,
            'kiloparsec': 30856775814913673000,
            'kpc': 30856775814913673000,
            'megaparsec': 30856775814913673000000,
            'mpc': 30856775814913673000000,
        }
    },
    'magnitude': {
        'base': 'magnitude_absolue',
        'unites': {
            'magnitude_absolue': 1,
            'magnitude_apparente': 1,
            'mag': 1,
        }
    },
    'masse_astronomique': {
        'base': 'kg',
        'unites': {
            'kg': 1,
            'kilogramme': 1,
            'masse_solaire': 1.98847e30,
            'masse_sol': 1.98847e30,
            'msol': 1.98847e30,
            'masse_terre': 5.97219e24,
            'mterre': 5.97219e24,
            'masse_jupiter': 1.89813e27,
            'mjup': 1.89813e27,
        }
    },
    'vitesse_vent': {
        'base': 'metre_par_seconde',
        'unites': {
            'metre_par_seconde': 1,
            'm/s': 1,
            'kilometre_par_heure': 0.277778,
            'km/h': 0.277778,
            'noeud': 0.514444,
            'kt': 0.514444,
            'mile_par_heure': 0.44704,
            'mph': 0.44704,
            'beaufort_1': 1,  # 1-3 km/h
            'beaufort_6': 11.94,  # 39-49 km/h
            'beaufort_12': 32.7,  # >118 km/h
        }
    },
    'precipitation': {
        'base': 'millimetre',
        'unites': {
            'millimetre': 1,
            'mm': 1,
            'centimetre': 10,
            'cm': 10,
            'pouce': 25.4,
            'in': 25.4,
            'litre_par_m2': 1,  # équivalent
            'l/m2': 1,
        }
    },
    'humidite': {
        'base': 'pourcentage',
        'unites': {
            'pourcentage': 1,
            '%': 1,
            'fraction': 100,
            'ratio': 100,
        }
    },
    'volume_cuisine': {
        'base': 'millilitre',
        'unites': {
            'millilitre': 1,
            'ml': 1,
            'litre': 1000,
            'l': 1000,
            'centilitre': 10,
            'cl': 10,
            'cuillere_cafe': 5,
            'c_cafe': 5,
            'cc': 5,
            'cuillere_soupe': 15,
            'c_soupe': 15,
            'cs': 15,
            'tasse': 240,
            'cup': 240,
            'verre': 200,
            'pinte_us': 473.176,
            'pinte': 473.176,
            'quart': 946.353,
            'gallon_us': 3785.41,
        }
    },
    'masse_cuisine': {
        'base': 'gramme',
        'unites': {
            'gramme': 1,
            'g': 1,
            'kilogramme': 1000,
            'kg': 1000,
            'once': 28.3495,
            'oz': 28.3495,
            'livre': 453.592,
            'lb': 453.592,
            'pincee': 0.5,
            'noix': 15,  # beurre
        }
    },
    'temperature_cuisine': {
        'base': 'celsius',
        'unites': {
            'celsius': 1,
            'c': 1,
            'thermostat_1': 30,
            'th1': 30,
            'thermostat_2': 60,
            'th2': 60,
            'thermostat_3': 90,
            'th3': 90,
            'thermostat_4': 120,
            'th4': 120,
            'thermostat_5': 150,
            'th5': 150,
            'thermostat_6': 180,
            'th6': 180,
            'thermostat_7': 210,
            'th7': 210,
            'thermostat_8': 240,
            'th8': 240,
        }
    },
    'rendement_agricole': {
        'base': 'kg_par_hectare',
        'unites': {
            'kg_par_hectare': 1,
            'kg/ha': 1,
            'tonne_par_hectare': 1000,
            't/ha': 1000,
            'quintal_par_hectare': 100,
            'q/ha': 100,
            'boisseau_par_acre': 62.77,  # blé
            'bu/ac': 62.77,
        }
    },
    'consommation_carburant': {
        'base': 'litre_par_100km',
        'unites': {
            'litre_par_100km': 1,
            'l/100km': 1,
            'litre_par_km': 100,
            'l/km': 100,
            'km_par_litre': -1,  # inverse (nécessite traitement spécial)
            'mpg_us': -235.215,  # miles par gallon (inverse)
            'mpg_uk': -282.481,
        }
    },
    'economie_energie': {
        'base': 'kwh_par_an',
        'unites': {
            'kwh_par_an': 1,
            'kwh/an': 1,
            'kwh_par_jour': 365,
            'kwh/jour': 365,
            'kwh_par_mois': 12,
            'kwh/mois': 12,
            'wh_par_jour': 0.365,
            'wh/jour': 0.365,
        }
    },
    'indice_refraction': {
        'base': 'sans_unite',
        'unites': {
            'sans_unite': 1,
            'indice': 1,
            'vide': 1.0,
            'air': 1.000293,
            'eau': 1.333,
            'verre': 1.52,
            'diamant': 2.42,
        }
    },
    'longueur_onde': {
        'base': 'nanometre',
        'unites': {
            'nanometre': 1,
            'nm': 1,
            'picometre': 0.001,
            'pm': 0.001,
            'angstrom': 0.1,
            'Å': 0.1,
            'micrometre': 1000,
            'µm': 1000,
            'millimetre': 1000000,
            'mm': 1000000,
        }
    },
    'token_ia': {
        'base': 'token',
        'unites': {
            'token': 1,
            'mot': 1.33,  # ~0.75 token par mot en moyenne
            'caractere': 0.25,  # ~4 caractères par token
            'kilotoken': 1000,
            'ktoken': 1000,
            'megatoken': 1000000,
            'mtoken': 1000000,
            'page': 400,  # ~400 tokens par page
            'livre': 100000,  # ~100k tokens par livre
        }
    },
    'devise': {
        'base': 'eur',
        'unites': {
            'eur': 1,
            'euro': 1,
            '€': 1,
            'usd': 1.08,  # approximation, varie
            'dollar': 1.08,
            '$': 1.08,
            'gbp': 0.86,  # livre sterling
            '£': 0.86,
            'jpy': 155,  # yen japonais
            '¥': 155,
            'chf': 0.93,  # franc suisse
            'cad': 1.46,  # dollar canadien
            'aud': 1.64,  # dollar australien
            'cny': 7.73,  # yuan chinois
            'inr': 90,  # roupie indienne
            'btc': 0.000024,  # bitcoin (très variable)
            'eth': 0.00036,  # ethereum (très variable)
        }
    },
    'niveau_sonore': {
        'base': 'decibel',
        'unites': {
            'decibel': 1,
            'db': 1,
            'dba': 1,  # pondéré A
            'dbc': 1,  # pondéré C
            'bel': 10,
            'neper': 8.686,
        }
    },
    'intensite_sonore': {
        'base': 'watt_par_m2',
        'unites': {
            'watt_par_m2': 1,
            'w/m²': 1,
            'microwatt_par_m2': 0.000001,
            'µw/m²': 0.000001,
        }
    },
    'format_papier': {
        'base': 'mm2',
        'unites': {
            'mm2': 1,
            'a0': 999949,  # 841 × 1189 mm
            'a1': 500024,  # 594 × 841 mm
            'a2': 250047,  # 420 × 594 mm
            'a3': 125010,  # 297 × 420 mm
            'a4': 62370,   # 210 × 297 mm
            'a5': 15540,   # 148 × 210 mm
            'letter': 60387,  # 8.5 × 11 pouces
            'legal': 78064,   # 8.5 × 14 pouces
            'tabloid': 120774, # 11 × 17 pouces
        }
    },
    'taille_ecran': {
        'base': 'pouce_diagonale',
        'unites': {
            'pouce_diagonale': 1,
            'pouce': 1,
            'inch': 1,
            '"': 1,
            'centimetre': 0.393701,
            'cm': 0.393701,
        }
    },
    'ratio_ecran': {
        'base': 'ratio',
        'unites': {
            'ratio': 1,
            '4_3': 1.333,    # ancien standard
            '16_9': 1.778,   # HD standard
            '16_10': 1.6,    # écrans PC
            '21_9': 2.333,   # ultra-wide
            '32_9': 3.556,   # super ultra-wide
            'cinemascope': 2.35,
        }
    },
    'pixel_densite': {
        'base': 'ppi',
        'unites': {
            'ppi': 1,  # pixels per inch
            'dpi': 1,
            'ppcm': 2.54,  # pixels per cm
            'retina': 326,  # densité Retina d'Apple
        }
    },
    'temps_geologique': {
        'base': 'annee',
        'unites': {
            'annee': 1,
            'an': 1,
            'siecle': 100,
            'millenaire': 1000,
            'million_annees': 1000000,
            'ma': 1000000,  # mega annum
            'milliard_annees': 1000000000,
            'ga': 1000000000,  # giga annum
            'eon': 1000000000,
        }
    },
    'echelle_temps_geologique': {
        'base': 'ma',
        'unites': {
            'ma': 1,
            'holocene': 0.0117,
            'pleistocene': 2.58,
            'pliocene': 5.333,
            'miocene': 23.03,
            'cretace': 145,
            'jurassique': 201.3,
            'trias': 251.9,
            'permien': 298.9,
            'cambrien': 541,
        }
    },
    'echelle_richter': {
        'base': 'magnitude',
        'unites': {
            'magnitude': 1,
            'richter': 1,
            'moment': 1,  # échelle de moment
        }
    },
    'age_biologique': {
        'base': 'annee_vie',
        'unites': {
            'annee_vie': 1,
            'mois_vie': 0.0833,
            'semaine_vie': 0.0192,
            'jour_vie': 0.00274,
        }
    },
    'sequence_adn': {
        'base': 'paire_bases',
        'unites': {
            'paire_bases': 1,
            'bp': 1,  # base pair
            'kilobase': 1000,
            'kb': 1000,
            'megabase': 1000000,
            'mb': 1000000,
            'gigabase': 1000000000,
            'gb': 1000000000,
            'gene': 10000,  # moyenne
            'chromosome_humain': 133000000,  # moyenne
            'genome_humain': 3200000000,
        }
    },
    'masse_moleculaire': {
        'base': 'dalton',
        'unites': {
            'dalton': 1,
            'da': 1,
            'uma': 1,  # unité de masse atomique
            'u': 1,
            'kilodalton': 1000,
            'kda': 1000,
            'megadalton': 1000000,
            'mda': 1000000,
        }
    },
    'vitesse_sedimentation': {
        'base': 'svedberg',
        'unites': {
            'svedberg': 1,
            's': 1,
            'millisvedberg': 0.001,
            'ms': 0.001,
        }
    },
    'dose_medicament': {
        'base': 'mg_par_kg',
        'unites': {
            'mg_par_kg': 1,
            'mg/kg': 1,
            'g_par_kg': 1000,
            'g/kg': 1000,
            'mcg_par_kg': 0.001,
            'µg/kg': 0.001,
        }
    },
    'glycemie': {
        'base': 'mmol_par_l',
        'unites': {
            'mmol_par_l': 1,
            'mmol/l': 1,
            'mg_par_dl': 0.0555,  # mg/dL
            'mg/dl': 0.0555,
            'g_par_l': 5.55,
            'g/l': 5.55,
        }
    },
    'pression_arterielle': {
        'base': 'mmhg',
        'unites': {
            'mmhg': 1,
            'mm_mercure': 1,
            'kpa': 7.50062,
            'cmh2o': 1.35951,  # cm d'eau
        }
    },
    'debit_sanguin': {
        'base': 'litre_par_min',
        'unites': {
            'litre_par_min': 1,
            'l/min': 1,
            'ml_par_min': 0.001,
            'ml/min': 0.001,
            'debit_cardiaque': 5,  # ~5 L/min au repos
        }
    },
    'energie_photon': {
        'base': 'electronvolt',
        'unites': {
            'electronvolt': 1,
            'ev': 1,
            'kiloelectronvolt': 1000,
            'kev': 1000,
            'megaelectronvolt': 1000000,
            'mev': 1000000,
            'gigaelectronvolt': 1000000000,
            'gev': 1000000000,
            'teraelectronvolt': 1000000000000,
            'tev': 1000000000000,
            'joule': 6.242e18,
        }
    },
    'masse_particule': {
        'base': 'ev_c2',
        'unites': {
            'ev_c2': 1,  # eV/c²
            'ev/c2': 1,
            'mev_c2': 1000000,
            'mev/c2': 1000000,
            'gev_c2': 1000000000,
            'gev/c2': 1000000000,
            'masse_electron': 510998.95,  # eV/c²
            'masse_proton': 938272088,  # eV/c²
        }
    },
    'section_efficace': {
        'base': 'barn',
        'unites': {
            'barn': 1,
            'b': 1,
            'millibarn': 0.001,
            'mb': 0.001,
            'microbarn': 0.000001,
            'µb': 0.000001,
            'metre_carre': 1e28,
        }
    },
    'luminosite_particule': {
        'base': 'cm2_s',
        'unites': {
            'cm2_s': 1,
            'cm⁻²s⁻¹': 1,
            'inverse_femtobarn_seconde': 1e-39,
            'fb⁻¹s⁻¹': 1e-39,
        }
    },
    'duree_vie_particule': {
        'base': 'seconde',
        'unites': {
            'seconde': 1,
            's': 1,
            'milliseconde': 0.001,
            'ms': 0.001,
            'microseconde': 0.000001,
            'µs': 0.000001,
            'nanoseconde': 0.000000001,
            'ns': 0.000000001,
            'picoseconde': 0.000000000001,
            'ps': 0.000000000001,
            'femtoseconde': 0.000000000000001,
            'fs': 0.000000000000001,
            'attoseconde': 0.000000000000000001,
            'as': 0.000000000000000001,
        }
    }
}


def trouver_categorie(unite):
    """
    Trouve la catégorie d'une unité
    
    Args:
        unite (str): Nom de l'unité
        
    Returns:
        str: Nom de la catégorie ou None si non trouvée
    """
    unite = unite.lower().strip()
    for categorie, donnees in UNITES.items():
        if unite in donnees['unites']:
            return categorie
    return None


def convertir(valeur, unite_depart, unite_arrivee):
    """
    Convertit une valeur d'une unité vers une autre
    
    Args:
        valeur (float): Valeur à convertir
        unite_depart (str): Unité de départ
        unite_arrivee (str): Unité d'arrivée
        
    Returns:
        float: Valeur convertie
        
    Raises:
        ValueError: Si les unités ne sont pas compatibles ou n'existent pas
    """
    unite_depart = unite_depart.lower().strip()
    unite_arrivee = unite_arrivee.lower().strip()
    
    # Gestion spéciale pour la température
    if unite_depart in ['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k']:
        return convertir_temperature(valeur, unite_depart, unite_arrivee)
    
    # Trouver les catégories
    categorie_depart = trouver_categorie(unite_depart)
    categorie_arrivee = trouver_categorie(unite_arrivee)
    
    if categorie_depart is None:
        raise ValueError(f"Unité de départ '{unite_depart}' non reconnue")
    
    if categorie_arrivee is None:
        raise ValueError(f"Unité d'arrivée '{unite_arrivee}' non reconnue")
    
    if categorie_depart != categorie_arrivee:
        raise ValueError(f"Impossible de convertir de {categorie_depart} vers {categorie_arrivee}")
    
    # Conversion: valeur -> unité de base -> unité d'arrivée
    facteur_depart = UNITES[categorie_depart]['unites'][unite_depart]
    facteur_arrivee = UNITES[categorie_arrivee]['unites'][unite_arrivee]
    
    valeur_base = valeur * facteur_depart
    resultat = valeur_base / facteur_arrivee
    
    return resultat


def convertir_temperature(valeur, unite_depart, unite_arrivee):
    """
    Convertit une température d'une unité vers une autre
    
    Args:
        valeur (float): Température à convertir
        unite_depart (str): Unité de départ (celsius/c, fahrenheit/f, kelvin/k)
        unite_arrivee (str): Unité d'arrivée
        
    Returns:
        float: Température convertie
    """
    unite_depart = unite_depart.lower().strip()
    unite_arrivee = unite_arrivee.lower().strip()
    
    # Normaliser les noms
    normalisation = {
        'c': 'celsius',
        'f': 'fahrenheit',
        'k': 'kelvin'
    }
    
    if unite_depart in normalisation:
        unite_depart = normalisation[unite_depart]
    if unite_arrivee in normalisation:
        unite_arrivee = normalisation[unite_arrivee]
    
    # Conversion vers Celsius d'abord
    if unite_depart == 'celsius':
        celsius = valeur
    elif unite_depart == 'fahrenheit':
        celsius = (valeur - 32) * 5/9
    elif unite_depart == 'kelvin':
        celsius = valeur - 273.15
    else:
        raise ValueError(f"Unité de température '{unite_depart}' non reconnue")
    
    # Conversion depuis Celsius vers l'unité d'arrivée
    if unite_arrivee == 'celsius':
        return celsius
    elif unite_arrivee == 'fahrenheit':
        return celsius * 9/5 + 32
    elif unite_arrivee == 'kelvin':
        return celsius + 273.15
    else:
        raise ValueError(f"Unité de température '{unite_arrivee}' non reconnue")


def afficher_unites_disponibles():
    """
    Affiche toutes les unités disponibles par catégorie
    """
    print("=" * 60)
    print("UNITÉS DISPONIBLES")
    print("=" * 60)
    
    for categorie, donnees in UNITES.items():
        print(f"\n{categorie.upper()}:")
        print(f"  Unité de base: {donnees['base']}")
        print(f"  Unités: {', '.join(sorted(donnees['unites'].keys()))}")
    
    print("\nTEMPÉRATURE:")
    print("  Unités: celsius, c, fahrenheit, f, kelvin, k")
    print("=" * 60)


def lister_unites_categorie(categorie):
    """
    Liste toutes les unités d'une catégorie
    
    Args:
        categorie (str): Nom de la catégorie
        
    Returns:
        list: Liste des unités disponibles
    """
    categorie = categorie.lower().strip()
    
    if categorie == 'temperature':
        return ['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k']
    
    if categorie in UNITES:
        return list(UNITES[categorie]['unites'].keys())
    
    return []


def lister_categories():
    """
    Liste toutes les catégories disponibles
    
    Returns:
        list: Liste des catégories
    """
    return list(UNITES.keys()) + ['temperature']


# Exemples d'utilisation
if __name__ == "__main__":
    print("Module de conversion d'unités\n")
    
    # Exemples de conversions
    exemples = [
        (100, 'km', 'mile', 'Distance'),
        (75, 'kg', 'lb', 'Masse'),
        (25, 'celsius', 'fahrenheit', 'Température'),
        (1000, 'watt', 'cv', 'Puissance'),
        (180, 'degre', 'rad', 'Angle'),
        (1024, 'mo', 'ko', 'Données'),
        (100, 'mbps', 'mo/s', 'Bande passante'),
        (8294400, 'pixel', '4k', 'Résolution'),
        (3600, 'mah', 'coulomb', 'Charge électrique'),
        (0.5, 'tesla', 'gauss', 'Champ magnétique'),
        (1, 'mole', 'mmol', 'Quantité de matière'),
        (1, 'ua', 'km', 'Distance astronomique'),
        (240, 'ml', 'cup', 'Volume cuisine'),
        (100, 'eur', 'usd', 'Devise'),
        (1000, 'token', 'page', 'Tokens IA'),
        (85, 'db', 'bel', 'Niveau sonore'),
        (62370, 'mm2', 'a4', 'Format papier'),
        (27, 'pouce', 'cm', 'Taille écran'),
        (1000, 'bp', 'kb', 'Séquence ADN'),
        (1, 'mev', 'ev', 'Énergie photon'),
    ]
    
    for valeur, unite_dep, unite_arr, description in exemples:
        try:
            resultat = convertir(valeur, unite_dep, unite_arr)
            print(f"{description}: {valeur} {unite_dep} = {resultat:.2f} {unite_arr}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    print("\n")
    afficher_unites_disponibles()
