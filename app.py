# Modules maths (optionnels): ne doivent pas empêcher le site de démarrer
try:
    from modules.maths.fonctions.stats import StatisticalAnalyzer
except Exception:
    StatisticalAnalyzer = None

try:
    from modules.maths.fonctions.multivariate import MultivariateAnalyzer
except Exception:
    MultivariateAnalyzer = None

try:
    from modules.maths.fonctions.spectral import SpectralAnalyzer  # type: ignore
except Exception:
    SpectralAnalyzer = None

# Modules plots (optionnels): catalogue + rendu Plotly
try:
    from modules.plots import get_plot_catalog as PLOTS_get_plot_catalog
    from modules.plots import make_figure_json as PLOTS_make_figure_json
except Exception:
    PLOTS_get_plot_catalog = None
    PLOTS_make_figure_json = None

# Modules data_manipulation (optionnels): Data Studio / Dashboard
try:
    from modules.data_manipulation import DataManipulator, FeatureSpec, MergeSpec  # type: ignore
except Exception:
    DataManipulator = None
    FeatureSpec = None
    MergeSpec = None

# Modules graph (optionnels): Graph Studio
try:
    from modules.graph import SUPPORTED_GRAPH_TYPES as GRAPH_SUPPORTED_TYPES  # type: ignore
    from modules.graph import build_graph_from_files as GRAPH_build_graph_from_files  # type: ignore
    from modules.graph import build_graph_from_inputs as GRAPH_build_graph_from_inputs  # type: ignore
    from modules.graph import export_graph_data as GRAPH_export_graph_data  # type: ignore
    from modules.graph import export_graph_text as GRAPH_export_graph_text  # type: ignore
    from modules.graph.graph_builder import (  # type: ignore
        InputItem as GRAPH_InputItem,
        compute_graph_metrics as GRAPH_compute_graph_metrics,
        filter_graph_by_degree as GRAPH_filter_graph_by_degree,
    )
except Exception:
    GRAPH_SUPPORTED_TYPES = None
    GRAPH_build_graph_from_files = None
    GRAPH_build_graph_from_inputs = None
    GRAPH_export_graph_data = None
    GRAPH_export_graph_text = None
    GRAPH_InputItem = None
    GRAPH_compute_graph_metrics = None
    GRAPH_filter_graph_by_degree = None
# Importation des biblios nécessaires ici
#===========================================================================
from flask import Flask, render_template, abort, redirect, url_for
from flask import request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import tempfile
import uuid
import json
# Gestion optionnelle de matplotlib
try:
    import matplotlib
    # Use a non-GUI backend to avoid creating NSWindow on macOS when running in a web server
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
import re
import pandas as pd
from pathlib import Path
import time

#===========================================================================
# speech2txt (optionnel: dépendances lourdes audio + whisper)
try:
    from modules.retranscription.speech2txt import AudioTranscriber  # type: ignore
except Exception:
    AudioTranscriber = None

_AUDIO_TRANSCRIBER_SINGLETON = None


def _get_audio_transcriber():
    global _AUDIO_TRANSCRIBER_SINGLETON
    if AudioTranscriber is None:
        return None
    if _AUDIO_TRANSCRIBER_SINGLETON is None:
        try:
            _AUDIO_TRANSCRIBER_SINGLETON = AudioTranscriber()
        except Exception:
            _AUDIO_TRANSCRIBER_SINGLETON = None
            return None
    return _AUDIO_TRANSCRIBER_SINGLETON
# Import du système de nettoyage modulaire (import résilient)
from modules.cleaner import Cleaner

try:
    from modules.cleaner import CSVEditor
except Exception:
    CSVEditor = None

# Import de ta classe FormatConverter depuis ton module converter.py
from modules.converter import FormatConverter

# Imports de traduction disponibles
try:
    from modules.translation.main import *
except ImportError:
    pass

# Modules disponibles (commentés car manquants)
# from modules.info import *
# from modules.traitement import *
# from modules.visualisation import *
# NOTE: on évite les imports globaux "*" ici (fragile + inutile pour démarrer)

#===========================================================================

PAGES = {
    "": "index.html",
    "fonctionnalites": "fonctionnalites.html",
    "confidentialite": "confidentialite.html",
    "mentions": "mentions.html",
    "contact": "contact.html",
    "apropos": "apropos.html",
    "docs": "docs.html",
    "biblio": "biblio.html",
    "dashboard": "dashboard.html",
    
    # Plateforme
    "lab": "plateform/lab.html",
    "traducteur": "plateform/logos.html",
    "audio": "plateform/audio.html",
    "nettoyeur": "plateform/saphir.html",
    
}
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-key-quantix-2025')

# Configuration de la session pour meilleure persistance
from datetime import timedelta
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # False en dev, True en prod avec HTTPS

# Dossier d'uploads (relatif au projet)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _safe_remove_file(path: str) -> bool:
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def _purge_session_uploads() -> int:
    """Supprime les fichiers uploadés associés à la session.

    Objectif: ne jamais conserver des données utilisateur entre pages.
    """
    removed = 0
    for key in ('uploaded_file', 'uploaded_file_right'):
        filename = session.pop(key, None)
        if not filename:
            continue
        # on ne supprime que des fichiers dans UPLOAD_FOLDER
        safe = os.path.basename(str(filename))
        path = os.path.join(UPLOAD_FOLDER, safe)
        removed += 1 if _safe_remove_file(path) else 0
    if removed:
        session.modified = True
    return removed


@app.before_request
def _privacy_purge_on_page_get():
    """Purge automatique lors d'une navigation (GET HTML).

    - On ne purge pas pour les endpoints API (sinon le Dashboard casserait).
    - On purge à chaque changement de page: aucune persistance de dataset.
    """
    try:
        if request.method != 'GET':
            return

        # Ne purge que pour les pages HTML (routes UI). Évite d'impacter des endpoints GET
        # de type API qui ne sont pas sous /api/.
        page_endpoints = {
            'home', 'fonctionnalites', 'confidentialite', 'mentions', 'contact', 'apropos', 'docs',
            'biblio', 'dashboard', 'essayer',
            'lab', 'graph',
            # Nouvelles plateformes (noms simplifiés)
            'traducteur', 'nettoyeur', 'audio',
            # Alias historiques (redirections)
            'logos', 'saphir', 'sonar', 'retranscripteur',
            'converter', 'labelling',
            'data_studio',
        }
        if request.endpoint not in page_endpoints:
            return

        # Purge uniquement lors d'une navigation effective entre pages.
        # - Pas de referrer -> première visite / entrée directe: on ne purge pas.
        # - Referrer = même page -> refresh: on ne purge pas.
        ref = request.referrer
        if not ref:
            return

        try:
            from urllib.parse import urlparse

            parsed = urlparse(ref)
            # Ignore referrers externes
            if parsed.netloc and request.host and parsed.netloc != request.host:
                return
            ref_path = parsed.path or ""
            if ref_path == request.path:
                return
        except Exception:
            # En cas de parsing foireux, on préfère ne pas purger.
            return

        _purge_session_uploads()
    except Exception:
        return


@app.route('/api/session/clear', methods=['POST'])
def api_session_clear():
    removed = _purge_session_uploads()
    return jsonify(success=True, removed=removed)
#===========================================================================
# HTML Pages Routes
# --- Pages principales ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fonctionnalites")
def fonctionnalites():
    return render_template("fonctionnalites.html")

@app.route("/confidentialite")
def confidentialite():
    return render_template("confidentialite.html")

@app.route("/mentions")
def mentions():
    return render_template("mentions.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/apropos")
def apropos():
    return render_template("apropos.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/autocleaner")
def autocleaner():
    return render_template("autocleaner.html")


#===========================================================================
COURSES_DIR = "templates/cours_html"

def clean_title(filename):
    # 1. remove extension
    name = filename.replace(".html", "")

    # 2. remove index numbers like "0007_"
    name = re.sub(r"^\d+[_-]?", "", name)

    # 3. replace underscores and hyphens by spaces
    name = re.sub(r"[_-]+", " ", name)

    # 4. capitalize each word
    name = name.strip().title()

    return name

@app.route("/cours/<path:filename>")
def afficher_cours(filename):
    # Rendre le template pour que Jinja2 traite les {{ url_for() }}
    try:
        return render_template(f"cours_html/{filename}")
    except:
        abort(404)

@app.route("/biblio")
def biblio():
    files = [f for f in os.listdir(COURSES_DIR) if f.endswith(".html")]

    courses = [
        {
            "file": f,
            "title": clean_title(f)
        }
        for f in files
    ]

    return render_template("biblio.html", courses=courses)

#===========================================================================

# --- Plateforme ---
@app.route("/lab")
def lab():
    # Vérifier si un fichier est en session et s'il est accessible
    filename = session.get('uploaded_file')
    file_available = False
    error_message = None
    
    if filename:
        path = os.path.join(UPLOAD_FOLDER, filename)
        
        if os.path.exists(path):
            file_available = True
        else:
            error_message = f"Fichier {filename} introuvable. Veuillez le télécharger à nouveau."
            session.pop('uploaded_file', None)
    else:
        error_message = "Aucun fichier chargé. Veuillez d'abord télécharger un fichier CSV."
    
    return render_template("plateform/lab.html", 
                          file_available=file_available,
                          filename=filename,
                          error_message=error_message)


@app.route("/data-studio")
def data_studio():
    # La page Data Studio a été remplacée par le Dashboard (Excel-like).
    # On conserve l'URL historique pour éviter un 404/500 en production.
    return redirect(url_for('dashboard'))


@app.route("/graph")
def graph():
    filename = session.get('uploaded_file')
    file_available = False
    error_message = None
    if filename:
        path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            file_available = True
        else:
            error_message = f"Fichier {filename} introuvable. Veuillez le télécharger à nouveau."

    return render_template(
        "plateform/graph.html",
        file_available=file_available,
        filename=filename,
        error_message=error_message,
    )

@app.route("/traducteur")
def traducteur():
    return render_template("plateform/logos.html")


@app.route("/nettoyeur")
def nettoyeur():
    return render_template("plateform/saphir.html")


@app.route("/retranscripteur")
def retranscripteur():
    return redirect(url_for('audio'))


# URLs historiques (redirections)
@app.route("/logos")
def logos():
    return redirect(url_for('traducteur'))


@app.route("/saphir")
def saphir():
    return redirect(url_for('nettoyeur'))


@app.route("/sonar")
def sonar():
    return redirect(url_for('audio'))


@app.route("/audio")
def audio():
    return render_template("plateform/audio.html")


@app.route("/converter")
def converter():
    return render_template("plateform/converter.html")


# Route de téléchargement pour fichiers traduits/traités
@app.route('/download/<path:filename>')
def download_file(filename):
    """Permet de télécharger les fichiers traités depuis le dossier uploads"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify(error=f"Fichier non trouvé: {str(e)}"), 404


# Page IA (informations)
@app.route('/ia')
def ia():
    return render_template('ia.html')



# Plateforme Labelling (Label Studio like)
@app.route('/labelling')
def labelling():
    return render_template('plateform/labelling.html')


@app.route('/labelling/annotate')
def labelling_annotate():
    doc_id = (request.args.get('doc_id') or '').strip()
    if not doc_id:
        return redirect(url_for('labelling'))
    return render_template('plateform/labelling_annotate.html', doc_id=doc_id)


# API Labelling (complète) via blueprint: /api/labelling/*
LABELLING_API_AVAILABLE = False
try:
    from modules.labelling.annotation import create_flask_blueprint  # type: ignore

    app.register_blueprint(create_flask_blueprint(url_prefix="/api/labelling"))
    LABELLING_API_AVAILABLE = True
except Exception as e:
    # On n'empêche pas le site de démarrer si labelling est indisponible.
    print(f"[LABELLING] Blueprint indisponible: {e}")


# --- Section Essayer ---
@app.route("/essayer")
def essayer():
    return render_template("essayer.html")


#===========================================================================
#Modules Routes
#===========================================================================

#speech2txt
@app.route("/api/audio/transcribe", methods=["POST"])
def api_audio_transcribe():
    transcriber = _get_audio_transcriber()
    if transcriber is None:
        return jsonify({"status": "error", "error": "Module audio indisponible sur ce serveur"}), 501

    if 'file' not in request.files:
        return jsonify({"status": "error", "error": "Aucun fichier fourni (champ 'file')"}), 400

    f = request.files['file']
    if not f or not getattr(f, 'filename', None):
        return jsonify({"status": "error", "error": "Fichier audio invalide"}), 400

    original_name = secure_filename(f.filename) or f"audio_{uuid.uuid4().hex}"
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.webm', '.mp4', '.mov'}:
        # fallback basé sur mimetype
        mt = (getattr(f, 'mimetype', None) or '').lower()
        if 'webm' in mt:
            suffix = '.webm'
        elif 'mpeg' in mt or 'mp3' in mt:
            suffix = '.mp3'
        elif 'wav' in mt:
            suffix = '.wav'
        elif 'mp4' in mt:
            suffix = '.mp4'
        else:
            suffix = '.webm'

    tmp_in = None
    try:
        tmp = tempfile.NamedTemporaryFile(prefix='audio_in_', suffix=suffix, delete=False)
        tmp_in = tmp.name
        tmp.close()
        f.save(tmp_in)

        out_format = (request.form.get('format', 'txt') or 'txt').strip().lower()
        if out_format == 'docx':
            out_docx = transcriber.transcribe(tmp_in, write_docx=True, write_txt=False)
            uploads_path = os.path.join(UPLOAD_FOLDER, os.path.basename(out_docx))
            if not os.path.exists(uploads_path):
                os.replace(out_docx, uploads_path)
            download_url = f'/uploads/{os.path.basename(out_docx)}'
            return jsonify({"status": "ok", "download_url": download_url})

        text = transcriber.transcribe(tmp_in, write_txt=False)
        return jsonify({"status": "ok", "text": text})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    finally:
        # Nettoyage best-effort des temporaires (input + wav converti)
        if tmp_in and os.path.exists(tmp_in):
            try:
                os.remove(tmp_in)
            except Exception:
                pass
        if tmp_in:
            wav = str(Path(tmp_in).with_suffix('.wav'))
            if os.path.exists(wav):
                try:
                    os.remove(wav)
                except Exception:
                    pass

#cleaner automatique intelligent
@app.route("/api/clean", methods=["POST"])
def api_clean():
    """Route pour le nettoyage automatique et intelligent des données."""
    try:
        start_time = time.time()

        # Gestion des fichiers uploadés ou chemin fourni
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            filepath = request.form.get("filepath")
            if not filepath or not os.path.exists(filepath):
                return jsonify({"success": False, "error": "Fichier non trouvé"}), 400

        def _parse_bool(value, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            if s in {"1", "true", "t", "yes", "y", "on", "oui"}:
                return True
            if s in {"0", "false", "f", "no", "n", "off", "non"}:
                return False
            return default

        def _parse_float(value, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        def _quality_score(df: pd.DataFrame) -> float:
            if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                return 0.0
            total_cells = df.shape[0] * df.shape[1]
            missing_ratio = float(df.isnull().sum().sum()) / float(total_cells)
            duplicate_ratio = float(df.duplicated().sum()) / float(max(df.shape[0], 1))
            score = 1.0 - missing_ratio - duplicate_ratio
            return max(0.0, min(1.0, score))
        
        # Paramètres de configuration
        preset = (request.form.get('preset') or '').strip().lower()  # legacy: web/scientific/business/survey
        output_format = request.form.get('output_format', 'csv')  # csv, excel, json
        
        # Configuration prédéfinie simplifiée (remplace PresetConfigs)
        preset_settings = {
            "web": {
                "trim_strings": True,
                "remove_duplicates": False,
                "normalize_spaces": True,
                "fix_typo": True,
                "auto_convert_types": True,
                "drop_missing": False,
                "fill_missing_values": False,
                "remove_outliers": False
            },
            "scientific": {
                "trim_strings": True,
                "remove_duplicates": False,  # Peut être important en maths
                "normalize_spaces": True,
                "auto_convert_types": True,
                "remove_outliers": False,  # Outliers peuvent être significatifs
                "drop_missing": False,
                "fill_missing_values": False
            },
            "business": {
                "trim_strings": True,
                "remove_duplicates": False,
                "normalize_spaces": True,
                "fix_typo": True,
                "auto_convert_types": True,
                "fill_missing_values": False,
                "drop_missing": False,
                "remove_outliers": False
            },
            "survey": {
                "trim_strings": True,
                "remove_duplicates": False,
                "normalize_spaces": True,
                "fix_typo": True,
                "auto_convert_types": True,
                "drop_missing": False,
                "fill_missing_values": False,
                "remove_outliers": False
            }
        }
        if preset in preset_settings:
            config_settings = preset_settings[preset].copy()
        else:
            # Mode simple (sans preset): defaults safe, orientés standardisation texte
            config_settings = {
                "trim_strings": True,
                "remove_duplicates": False,
                "normalize_spaces": True,
                "fix_typo": True,
                "auto_convert_types": True,
                "drop_missing": False,
                "fill_missing_values": False,
                "remove_outliers": False,
                "remove_empty": True,
                "lowercase_strings": True,
                "normalize_text": False,
            }
        
        # Personnalisation optionnelle des paramètres
        config_settings['auto_convert_types'] = _parse_bool(request.form.get('auto_detect_types'), config_settings.get('auto_convert_types', False))
        config_settings['fix_typo'] = _parse_bool(request.form.get('correct_typos'), config_settings.get('fix_typo', True))
        config_settings['remove_outliers'] = _parse_bool(request.form.get('detect_outliers'), config_settings.get('remove_outliers', False))

        # Toggles explicites (UI cases à cocher)
        if request.form.get('trim_strings') is not None:
            config_settings['trim_strings'] = _parse_bool(request.form.get('trim_strings'), config_settings.get('trim_strings', True))
        if request.form.get('normalize_spaces') is not None:
            config_settings['normalize_spaces'] = _parse_bool(request.form.get('normalize_spaces'), config_settings.get('normalize_spaces', True))
        if request.form.get('remove_empty') is not None:
            config_settings['remove_empty'] = _parse_bool(request.form.get('remove_empty'), config_settings.get('remove_empty', True))
        if request.form.get('lowercase_strings') is not None:
            config_settings['lowercase_strings'] = _parse_bool(request.form.get('lowercase_strings'), config_settings.get('lowercase_strings', False))
        if request.form.get('normalize_text') is not None:
            # normalisation avancée (accents/emojis)
            config_settings['normalize_text'] = _parse_bool(request.form.get('normalize_text'), config_settings.get('normalize_text', False))

        # Options avancées (si fournies)
        # (on garde des défauts sûrs pour ne pas être destructif)
        if request.form.get('drop_missing') is not None:
            config_settings['drop_missing'] = _parse_bool(request.form.get('drop_missing'), True)
        if request.form.get('remove_duplicates') is not None:
            config_settings['remove_duplicates'] = _parse_bool(request.form.get('remove_duplicates'), True)
        if request.form.get('fill_missing_values') is not None:
            config_settings['fill_missing_values'] = _parse_bool(request.form.get('fill_missing_values'), False)

        if request.form.get('missing_threshold') is not None:
            config_settings['missing_threshold'] = _parse_float(request.form.get('missing_threshold'), 0.3)
        if request.form.get('missing_axis') is not None:
            ax = str(request.form.get('missing_axis')).strip().lower()
            if ax in {'x', 'y'}:
                config_settings['missing_axis'] = ax

        if request.form.get('duplicate_keep') is not None:
            config_settings['duplicate_keep'] = str(request.form.get('duplicate_keep'))

        if request.form.get('outlier_method') is not None:
            method = str(request.form.get('outlier_method')).strip().lower()
            if method in {'iqr', 'zscore'}:
                config_settings['outlier_method'] = method
        if request.form.get('outlier_threshold') is not None:
            config_settings['outlier_threshold'] = _parse_float(request.form.get('outlier_threshold'), 1.5)

        if request.form.get('fill_strategy') is not None:
            config_settings['fill_strategy'] = str(request.form.get('fill_strategy'))
        if request.form.get('fill_constant') is not None:
            config_settings['fill_constant'] = request.form.get('fill_constant')

        # Conversion d'unités (opt-in)
        if request.form.get('convert_units') is not None:
            config_settings['convert_units'] = _parse_bool(request.form.get('convert_units'), False)
        if request.form.get('unit_mode') is not None:
            mode = str(request.form.get('unit_mode')).strip().lower()
            if mode in {'add', 'split', 'replace'}:
                config_settings['unit_mode'] = mode
        if request.form.get('unit_parse_threshold') is not None:
            config_settings['unit_parse_threshold'] = _parse_float(request.form.get('unit_parse_threshold'), 0.6)
        
        # Charger une version brute pour score qualité "avant"
        cleaner_probe = Cleaner(drop_missing=False, remove_duplicates=False)
        df_before = cleaner_probe.load_file(filepath)
        quality_before = _quality_score(df_before)

        # Nettoyage avec Cleaner (pipeline modulaire)
        cleaner = Cleaner(**config_settings)
        cleaned_df = cleaner.clean(file_path=filepath)
        quality_after = _quality_score(cleaned_df)
        
        # Rapport
        stats = cleaner.get_stats(detailed=True)
        operations = stats.get('operations', []) or []

        # Rapport de transformations structuré (JSON-friendly)
        transformation_report = {}
        try:
            transformation_report = cleaner.get_transformation_report()
        except Exception:
            transformation_report = {}

        warnings_list = []
        if quality_after < quality_before:
            warnings_list.append("Le score qualité a diminué (vérifier les options de nettoyage).")
        if stats.get('rows_before', 0) and stats.get('rows_after', 0) < stats.get('rows_before', 0) * 0.5:
            warnings_list.append("Beaucoup de lignes ont été supprimées (drop_missing/remove_outliers).")
        
        # Génération du nom de fichier de sortie
        input_path = Path(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_path.stem}_cleaned_{timestamp}"
        
        # Sauvegarde selon le format demandé
        if output_format == 'excel':
            output_file = f"{output_name}.xlsx"
            output_path = os.path.join(UPLOAD_FOLDER, output_file)
            cleaned_df.to_excel(output_path, index=False)
        elif output_format == 'json':
            output_file = f"{output_name}.json"
            output_path = os.path.join(UPLOAD_FOLDER, output_file)
            cleaned_df.to_json(output_path, orient='records', indent=2)
        elif output_format == 'parquet':
            output_file = f"{output_name}.parquet"
            output_path = os.path.join(UPLOAD_FOLDER, output_file)
            cleaned_df.to_parquet(output_path, index=False)
        else:  # csv par défaut
            output_file = f"{output_name}.csv"
            output_path = os.path.join(UPLOAD_FOLDER, output_file)
            cleaned_df.to_csv(output_path, index=False, na_rep="")

        # Export du rapport JSON (à côté du fichier nettoyé)
        report_file = f"{output_name}_report.json"
        report_path = os.path.join(UPLOAD_FOLDER, report_file)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(transformation_report, f, ensure_ascii=False, indent=2, default=str)
        except Exception as _e:
            # Ne jamais faire échouer le nettoyage juste pour l'export du rapport
            report_file = None
        
        processing_time = time.time() - start_time

        # Préparer la réponse attendue par le frontend (saphir.html)
        response_data = {
            "success": True,
            "message": "Nettoyage automatique terminé avec succès",
            "output_file": output_file,
            "download_url": f"/download/{output_file}",
            "report_file": report_file,
            "report_download_url": f"/download/{report_file}" if report_file else None,
            "transformation_report": transformation_report,
            "original_shape": [int(stats.get('rows_before', 0)), int(stats.get('cols_before', 0))],
            "final_shape": [int(stats.get('rows_after', 0)), int(stats.get('cols_after', 0))],
            "operations_count": len(operations),
            "operations": operations,
            "quality_improvement": {
                "before": f"{quality_before * 100:.1f}%",
                "after": f"{quality_after * 100:.1f}%",
                "improvement": f"{(quality_after - quality_before) * 100:+.1f}%",
            },
            "processing_time": f"{processing_time:.2f}s",
            "warnings": warnings_list,
            "preset_used": preset if preset in preset_settings else "custom"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Erreur lors du nettoyage: {str(e)}"
        }), 500


@app.route("/api/clean/preview", methods=["POST"])
def api_clean_preview():
    """Aperçu du nettoyage sans sauvegarde pour validation utilisateur."""
    try:
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            filepath = request.form.get("filepath")
            if not filepath or not os.path.exists(filepath):
                return jsonify({"success": False, "error": "Fichier non trouvé"}), 400
        
        # Chargement du dataset (multi-format)
        cleaner_temp = Cleaner(drop_missing=False, remove_duplicates=False)
        df = cleaner_temp.load_file(filepath)
        
        total_nulls = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        if df.shape[0] == 0 or df.shape[1] == 0:
            quality_score = 0.0
        else:
            quality_score = max(
                0.0,
                1.0 - (total_nulls / (df.shape[0] * df.shape[1])) - (duplicates / df.shape[0])
            )
        
        # Détection des types de colonnes et problèmes potentiels
        column_analysis = []
        for col in df.columns:
            row_count = len(df)
            analysis = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / row_count) * 100) if row_count else 0.0,
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            # Détection du type probable avec Cleaner (sans modifier df)
            if df[col].dtype == 'object':
                analysis["detected_type"] = cleaner_temp.detect_column_type(df[col]).get('type', 'unknown')
            else:
                analysis["detected_type"] = str(df[col].dtype)
            
            column_analysis.append(analysis)
        
        return jsonify({
            "success": True,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "quality_score": f"{quality_score:.2%}",
            "total_nulls": int(total_nulls),
            "duplicates": int(duplicates),
            "columns": column_analysis,
            "recommendations": {
                "suggested_preset": "business",  # Logique de suggestion à améliorer
                "critical_issues": [
                    issue for issue in [
                        "Nombreuses valeurs manquantes" if ((total_nulls / (df.shape[0] * df.shape[1])) if (df.shape[0] and df.shape[1]) else 0.0) > 0.2 else None,
                        "Doublons détectés" if duplicates > 0 else None,
                        "Qualité faible" if quality_score < 0.5 else None
                    ] if issue
                ]
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Erreur lors de l'analyse: {str(e)}"
        }), 500

#info
@app.route("/api/info", methods=["POST"])
def api_info():
    try:
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        else:
            filepath = request.form.get("filepath")
            if not filepath or not os.path.exists(filepath):
                return jsonify({'error': 'Fichier non trouvé'}), 400

        # Utiliser Cleaner pour analyser le fichier (remplace FileAnalyzer)
        cleaner = Cleaner()
        df = cleaner.load_file(filepath)
        
        analysis = {
            'filename': os.path.basename(filepath),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'duplicates': int(df.duplicated().sum()),
            'file_size': int(os.path.getsize(filepath)) if os.path.exists(filepath) else 0
        }
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

#traduction
import threading
import time

# Dictionnaire pour stocker la progression des traductions
translation_progress = {}

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Route pour traduire des documents multiformat.
    Accepte les fichiers via upload et retourne un ID de tâche.
    """
    if 'file' not in request.files:
        return jsonify(success=False, error='Aucun fichier fourni'), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='Nom de fichier vide'), 400
    
    # Paramètres de traduction
    target_lang = request.form.get('target_lang', 'fr')
    preserve_foreign_quotes = request.form.get('preserve_foreign_quotes', 'true').lower() == 'true'
    # NOTE: Docling/Granite est désactivé (fonctionnalité en développement).
    # On conserve le champ `use_docling` côté front pour compatibilité, mais on l'ignore.
    use_docling = False
    
    try:
        # Sauvegarder le fichier uploadé
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        # Créer le nom du fichier de sortie
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_traduit_{target_lang}{ext}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # Générer un ID unique pour cette traduction
        task_id = f"{int(time.time())}_{filename}"
        
        # Initialiser la progression
        translation_progress[task_id] = {
            'status': 'processing',
            'current_page': 0,
            'total_pages': 0,
            'output_path': None,
            'error': None,
            'should_stop': False,
            'output_file': output_path,
            'elapsed_seconds': 0.0,
            'avg_per_page': None
        }

        # Fonction callback pour la progression
        def progress_callback(current_page, total_pages, elapsed_seconds=None, avg_per_page=None):
            translation_progress[task_id]['current_page'] = current_page
            translation_progress[task_id]['total_pages'] = total_pages
            if elapsed_seconds is not None:
                translation_progress[task_id]['elapsed_seconds'] = elapsed_seconds
            if avg_per_page is not None:
                translation_progress[task_id]['avg_per_page'] = avg_per_page
            # Retourner True si on doit continuer, False si on doit arrêter
            return not translation_progress[task_id]['should_stop']
        
        # Fonction de traduction dans un thread
        def translate_async():
            try:
                from modules.translation.main import translate
                
                # Traduire le document
                translate(
                    input_path=input_path,
                    output_path=output_path,
                    target_lang=target_lang,
                    preserve_foreign_quotes=preserve_foreign_quotes,
                    use_docling=use_docling,
                    progress_callback=progress_callback
                )
                
                # Vérifier si arrêté ou complété
                if translation_progress[task_id]['should_stop']:
                    translation_progress[task_id]['status'] = 'stopped'
                    # Si le fichier partiel existe, on le garde
                    if os.path.exists(output_path):
                        translation_progress[task_id]['output_path'] = output_filename
                else:
                    translation_progress[task_id]['status'] = 'completed'
                    translation_progress[task_id]['output_path'] = output_filename
                
            except Exception as e:
                print(f"Erreur traduction: {e}")
                translation_progress[task_id]['status'] = 'error'
                translation_progress[task_id]['error'] = str(e)
                # Même en cas d'erreur, garder le fichier partiel s'il existe
                if os.path.exists(output_path):
                    translation_progress[task_id]['output_path'] = output_filename
        
        # Lancer la traduction dans un thread
        thread = threading.Thread(target=translate_async)
        thread.daemon = True
        thread.start()
        
        # Retourner l'ID de tâche immédiatement
        return jsonify(
            success=True,
            task_id=task_id,
            message='Traduction en cours'
        )
        
    except Exception as e:
        print(f"Erreur traduction: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/translate/table-columns", methods=["POST"])
def api_translate_table_columns():
    """Retourne les colonnes disponibles d'un fichier tabulaire (CSV/XLSX/TSV).

    Objectif: permettre au front (Logos) de proposer automatiquement les colonnes traduisibles.
    Le fichier est lu en mémoire, limité à un échantillon de lignes pour rester rapide.
    """

    if 'file' not in request.files:
        return jsonify(success=False, error='Aucun fichier fourni'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='Nom de fichier vide'), 400

    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    ext = (ext or '').lower().lstrip('.')

    sample_rows = 200
    raw = file.read()

    try:
        import io

        df = None

        if ext in {'csv', 'tsv'}:
            buf = io.BytesIO(raw)
            if ext == 'tsv':
                df = pd.read_csv(buf, dtype=str, keep_default_na=False, nrows=sample_rows, sep='\t')
            else:
                # sep=None + engine=python => sniff simple du séparateur
                df = pd.read_csv(buf, dtype=str, keep_default_na=False, nrows=sample_rows, sep=None, engine='python')

        elif ext in {'xlsx', 'xls'}:
            buf = io.BytesIO(raw)
            df = pd.read_excel(buf, dtype=str, nrows=sample_rows)
            # Harmonise les NaN excel
            df = df.fillna('')

        elif ext in {'parquet'}:
            # Pandas requiert généralement pyarrow/fastparquet.
            # On passe par un fichier temporaire pour maximiser la compatibilité.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                try:
                    df = pd.read_parquet(tmp.name)
                except Exception as e:
                    return jsonify(success=False, error=f"Lecture parquet impossible (installe pyarrow): {e}"), 400
            # échantillonnage
            if df is not None and len(df) > sample_rows:
                df = df.head(sample_rows)
            # tout en string pour les heuristiques
            df = df.astype(str).fillna('')

        else:
            return jsonify(success=False, error=f"Format non supporté pour colonnes: .{ext}"), 400

        if df is None or df.shape[1] == 0:
            return jsonify(success=False, error='Aucune colonne détectée'), 400

        col_infos = []
        suggestion_candidates = []

        bad_name_re = re.compile(r"(^|_)(id|uuid|guid|code|ref|reference|num|numero|n°|date|timestamp|time|prix|price|amount|montant|score)(_|$)", re.IGNORECASE)
        alpha_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")

        for idx, col in enumerate(df.columns.tolist()):
            name = str(col)
            s = df[col]
            # vide si NaN ou string vide/espaces
            s2 = s.astype(str)
            stripped = s2.str.strip()
            empty_mask = stripped.eq('') | stripped.eq('nan') | stripped.eq('none')
            empty_ratio = float(empty_mask.mean()) if len(stripped) else 0.0
            non_empty = stripped[~empty_mask]

            samples = []
            if len(non_empty):
                samples = non_empty.drop_duplicates().head(3).tolist()

            avg_len = float(non_empty.str.len().mean()) if len(non_empty) else 0.0
            # ratio valeurs numériques
            numeric_ratio = 0.0
            if len(non_empty):
                numeric_ratio = float(pd.to_numeric(non_empty, errors='coerce').notna().mean())
            # ratio valeurs contenant des lettres
            alpha_ratio = 0.0
            if len(non_empty):
                alpha_ratio = float(non_empty.apply(lambda v: bool(alpha_re.search(v))).mean())

            col_infos.append(
                {
                    'name': name,
                    'index': idx,
                    'empty_ratio': round(empty_ratio, 4),
                    'non_empty_ratio': round(1.0 - empty_ratio, 4),
                    'avg_len': round(avg_len, 2),
                    'numeric_ratio': round(numeric_ratio, 4),
                    'alpha_ratio': round(alpha_ratio, 4),
                    'samples': samples,
                }
            )

            # heuristique "colonne texte" traduisible
            if bad_name_re.search(name):
                continue
            if (1.0 - empty_ratio) < 0.1:
                continue
            if alpha_ratio < 0.2:
                continue
            if numeric_ratio > 0.6:
                continue
            score = (1.0 - empty_ratio) * min(1.0, avg_len / 20.0) * alpha_ratio * (1.0 - numeric_ratio)
            suggestion_candidates.append((score, name))

        suggestions = [n for _, n in sorted(suggestion_candidates, key=lambda t: t[0], reverse=True)[:8]]

        return jsonify(success=True, columns=col_infos, suggested=suggestions)

    except Exception as e:
        print(f"Erreur analyse colonnes tabulaires: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/translate/table-column", methods=["POST"])
def api_translate_table_column():
    """Traduit une colonne précise d'un fichier tabulaire.

    Accepte un upload de fichier et retourne un fichier de sortie téléchargeable.

    Form fields:
      - target_lang (default: 'fr')
      - source_lang (default: 'auto')
      - column (nom de colonne) OU column_index (index 0-based)
      - replace (true/false, default false)
      - output_column (optionnel)
      - encoding/delimiter/quotechar (optionnels pour CSV)
    """
    if 'file' not in request.files:
        return jsonify(success=False, error='Aucun fichier fourni'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='Nom de fichier vide'), 400

    target_lang = request.form.get('target_lang', 'fr')
    source_lang = request.form.get('source_lang', 'auto')
    replace = request.form.get('replace', 'false').lower() == 'true'
    output_column = request.form.get('output_column')

    column = request.form.get('column')
    column_index = request.form.get('column_index')
    if (not column) and (column_index is None or str(column_index).strip() == ''):
        return jsonify(success=False, error="'column' ou 'column_index' est requis"), 400

    # CSV options
    encoding = request.form.get('encoding', 'utf-8')
    delimiter = request.form.get('delimiter')
    quotechar = request.form.get('quotechar')

    try:
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_col_traduit_{target_lang}{ext}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

        if column_index is not None and str(column_index).strip() != '':
            try:
                column_arg = int(column_index)
            except Exception:
                return jsonify(success=False, error="column_index doit être un entier"), 400
        else:
            column_arg = str(column)

        from modules.translation.main import translate_table_column

        report = translate_table_column(
            input_path,
            output_path,
            column=column_arg,
            target_lang=target_lang,
            source_lang=source_lang,
            output_column=output_column,
            replace=replace,
            encoding=encoding,
            delimiter=delimiter,
            quotechar=quotechar,
        )

        return jsonify(
            success=True,
            output_path=output_filename,
            download_url=f"/download/{output_filename}",
            report=report,
        )

    except Exception as e:
        print(f"Erreur traduction colonne tabulaire: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/translate/table-columns-batch", methods=["POST"])
def api_translate_table_columns_batch():
    """Traduit plusieurs colonnes d'un fichier tabulaire en une seule passe.

    Form fields:
      - target_lang (default: 'fr')
      - source_lang (default: 'auto')
      - columns (JSON array string) OU columns[] (multi)
      - replace (true/false, default false)
      - output_suffix (optionnel, ex: "translated_en")
      - encoding/delimiter/quotechar (optionnels pour CSV)
    """
    if 'file' not in request.files:
        return jsonify(success=False, error='Aucun fichier fourni'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='Nom de fichier vide'), 400

    target_lang = request.form.get('target_lang', 'fr')
    source_lang = request.form.get('source_lang', 'auto')
    replace = request.form.get('replace', 'false').lower() == 'true'
    output_suffix = request.form.get('output_suffix')

    # CSV options
    encoding = request.form.get('encoding', 'utf-8')
    delimiter = request.form.get('delimiter')
    quotechar = request.form.get('quotechar')

    # Parse columns
    cols = []
    try:
        raw_cols = request.form.get('columns')
        if raw_cols:
            import json

            parsed = json.loads(raw_cols)
            if isinstance(parsed, list):
                cols = parsed
        if not cols:
            cols = request.form.getlist('columns[]')
    except Exception:
        cols = request.form.getlist('columns[]')

    cols = [c for c in cols if c is not None and str(c).strip() != '']
    if not cols:
        return jsonify(success=False, error='columns is required'), 400

    # normalise: ints si possible
    normalized_cols = []
    for c in cols:
        s = str(c).strip()
        try:
            normalized_cols.append(int(s))
        except Exception:
            normalized_cols.append(s)

    try:
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_cols_traduit_{target_lang}{ext}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

        from modules.translation.main import translate_table_columns

        report = translate_table_columns(
            input_path,
            output_path,
            columns=normalized_cols,
            target_lang=target_lang,
            source_lang=source_lang,
            replace=replace,
            output_suffix=output_suffix,
            encoding=encoding,
            delimiter=delimiter,
            quotechar=quotechar,
        )

        return jsonify(
            success=True,
            output_path=output_filename,
            download_url=f"/download/{output_filename}",
            report=report,
        )
    except Exception as e:
        print(f"Erreur traduction multi-colonnes: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/api/translate/progress/<task_id>", methods=["GET"])
def get_translation_progress(task_id):
    """
    Récupère la progression d'une traduction.
    """
    if task_id not in translation_progress:
        # Cas fréquent: serveur relancé / dict en mémoire perdu / tâche expirée.
        # On renvoie success=True pour que le front puisse afficher un message propre.
        return jsonify(
            success=True,
            status='missing',
            current_page=0,
            total_pages=0,
            output_path=None,
            error=None,
            message='Tâche introuvable (serveur relancé ou tâche expirée). Relancez la traduction.',
            elapsed_seconds=0,
            avg_per_page=None,
        )
    
    progress = translation_progress[task_id]
    
    # Nettoyer les tâches terminées après 5 minutes
    if progress['status'] in ['completed', 'error']:
        # On pourrait ajouter un nettoyage automatique ici
        pass
    
    return jsonify(
        success=True,
        status=progress['status'],
        current_page=progress['current_page'],
        total_pages=progress['total_pages'],
        output_path=progress['output_path'],
        error=progress['error'],
        elapsed_seconds=progress.get('elapsed_seconds', 0),
        avg_per_page=progress.get('avg_per_page')
    )


@app.route("/api/translate/stop/<task_id>", methods=["POST"])
def stop_translation(task_id):
    """
    Arrête une traduction en cours.
    """
    if task_id not in translation_progress:
        # Même logique: éviter un popup "Tâche introuvable" côté UI.
        return jsonify(
            success=True,
            status='missing',
            message='Impossible d\'arrêter: tâche introuvable (serveur relancé ou tâche expirée).',
        )
    
    progress = translation_progress[task_id]
    
    if progress['status'] == 'processing':
        progress['should_stop'] = True
        return jsonify(
            success=True,
            message='Arrêt demandé, sauvegarde du fichier partiel en cours...'
        )
    else:
        # Ne pas traiter comme une "erreur" côté UI: la tâche n'est juste plus stoppable.
        return jsonify(
            success=True,
            status=str(progress.get('status')),
            message=f"La tâche est déjà {progress.get('status')}.",
        )



# visaulisation
@app.route("/api/visualize", methods=["POST"])
def api_visualize():
    filepath = request.form["filepath"]
    df = pd.read_csv(filepath)
    cleaner = Cleaner(
        remove_empty=True,
        remove_duplicates=True,
        trim_strings=True
    )
    cleaned_df = cleaner.clean(df)
    
    # Générer des visualisations basiques (Visualizer non disponible)
    # viz = Visualizer(cleaned_df)
    # plots = viz.generate_all_plots("static/plots")
    
    return jsonify({"status": "ok", "message": "Données nettoyées avec succès"})


# ===== Convenience / shim endpoints expected by the frontend templates =====
@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False, error='No file part'), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify(success=False, error='No selected file'), 400
    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_FOLDER, filename)
    f.save(dest)

    slot = (request.form.get('slot') or request.args.get('slot') or 'primary').strip().lower()
    # store in session for subsequent actions
    if slot in {'right', 'secondary', 'second'}:
        session['uploaded_file_right'] = filename
    else:
        session['uploaded_file'] = filename
    session.modified = True  # Force la sauvegarde de la session
    
    file_info = {'name': filename, 'path': dest, 'uploaded_at': datetime.utcnow().isoformat(), 'slot': slot}
    download_url = f'/uploads/{filename}'
    return jsonify(success=True, file_info=file_info, download_url=download_url)


# Endpoint legacy: certaines pages historiques utilisent encore /upload
@app.route('/upload', methods=['POST'])
def upload_legacy():
    if 'file' not in request.files:
        return jsonify(success=False, message='Aucun fichier fourni'), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify(success=False, message='Nom de fichier vide'), 400

    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_FOLDER, filename)
    f.save(dest)

    # Comportement historique: un seul slot (primary)
    session['uploaded_file'] = filename
    session.modified = True

    return jsonify(success=True, message='Fichier uploadé avec succès', path=dest, filename=filename, download_url=f'/uploads/{filename}')


# Conversion de format
@app.route('/api/convert/capabilities', methods=['GET'])
def api_convert_capabilities():
    """Expose les conversions réellement disponibles.

    Important: cette liste doit rester alignée avec le backend (routeur).
    Le frontend doit s'appuyer dessus pour ne jamais proposer des couples inexistants.
    """
    try:
        from modules.converter import CONVERSION_MAP

        # Filtrer les couples réellement utilisables sur ce serveur.
        # Objectif: ne pas afficher de conversions "fantômes" si une dépendance manque.
        mapping = {}
        for (src, dst), (module_name, func_name) in CONVERSION_MAP.items():
            try:
                mod = __import__(f"modules.converter.{module_name}", fromlist=[func_name])
                func = getattr(mod, func_name, None)
                if func is None:
                    continue

                # Cas spéciaux: image2image nécessite Pillow réellement présent
                if module_name == "image2image":
                    if getattr(mod, "Image", None) is None:
                        continue

                mapping.setdefault(str(src), set()).add(str(dst))
            except Exception:
                # Module non importable (dépendance manquante, erreur import, etc.)
                continue

        # sérialisation
        mapping_out = {k: sorted(v) for k, v in sorted(mapping.items(), key=lambda kv: kv[0])}
        all_sources = sorted(mapping_out.keys())
        all_targets = sorted({dst for v in mapping_out.values() for dst in v})

        return jsonify(
            success=True,
            conversions=mapping_out,
            sources=all_sources,
            targets=all_targets,
            aliases={
                "excel": "xlsx",
                "xls": "xlsx",
                "md": "markdown",
                "jpeg": "jpg",
            },
        )
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/convert', methods=['POST'])
def api_convert():
    if 'file' not in request.files:
        return jsonify(success=False, error='Aucun fichier fourni'), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify(success=False, error='Nom de fichier vide'), 400

    # Compat UI: Saphir envoie `output_format` (csv/excel/json)
    target_format = (
        request.form.get('target_format')
        or request.form.get('output_format')
        or request.form.get('target')
    )
    if not target_format:
        return jsonify(success=False, error='Format cible non précisé'), 400

    # Optionnel: format source forcé
    source_format = request.form.get('source_format')

    def _parse_bool(v, default=False):
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if s in ('0', 'false', 'no', 'n', 'off'):
            return False
        return default

    try:
        start_time = time.time()
        filename = secure_filename(f.filename)
        base, ext = os.path.splitext(filename)
        unique = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stored_name = f"{base}_{unique}{ext}" if ext else f"{base}_{unique}"
        dest = os.path.join(UPLOAD_FOLDER, stored_name)
        f.save(dest)

        # Déduire le format source (ou prendre celui fourni)
        src_format = (source_format or Path(stored_name).suffix).lower().lstrip('.')

        # Options conversion (facultatives)
        delimiter = request.form.get('delimiter')
        table_index = request.form.get('table_index')
        combine_tables = _parse_bool(request.form.get('combine_tables'), default=False)
        multi_tables = _parse_bool(request.form.get('multi_tables'), default=False)

        conv_kwargs = {}
        if delimiter:
            conv_kwargs['delimiter'] = delimiter
        if table_index is not None and str(table_index).strip() != '':
            conv_kwargs['table_index'] = int(table_index)
        if combine_tables:
            conv_kwargs['combine_tables'] = True
        if multi_tables:
            conv_kwargs['multi_tables'] = True

        # Par défaut, pour PDF→CSV, on préfère un seul CSV (table 0)
        # (multi_tables=true permet de récupérer toutes les tables en zip).
        if src_format == 'pdf' and str(target_format).lower().lstrip('.') in ('csv',) and not multi_tables:
            conv_kwargs.setdefault('table_index', 0)

        # Utilise le routeur universel
        from modules.converter.convert_router import convert_any_to_any

        result = convert_any_to_any(
            dest,
            src_format=src_format,
            dst_format=target_format,
            output_dir=UPLOAD_FOLDER,
            **conv_kwargs,
        )

        # Normaliser en un fichier téléchargeable
        output_path = None
        warnings_list = []

        # Multi-sorties: dict/list -> zip
        if isinstance(result, dict):
            # cas typique: pdf->csv tables
            output_dir = result.get('output_dir') or result.get('output_folder')
            if output_dir:
                import zipfile

                zip_name = f"{Path(dest).stem}_converted_{int(time.time())}.zip"
                zip_path = Path(UPLOAD_FOLDER) / zip_name
                base = Path(output_dir)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                    for p in base.rglob('*'):
                        if p.is_file():
                            z.write(p, arcname=str(p.relative_to(base)))
                output_path = str(zip_path)
                warnings_list.append("Conversion multi-fichiers: résultat fourni en .zip")
            else:
                raise RuntimeError("Conversion multi-sorties non récupérable (output_dir manquant)")

        elif isinstance(result, (list, tuple)):
            import zipfile

            files = [Path(p) for p in result if p]
            if not files:
                raise RuntimeError("Conversion n'a produit aucun fichier")
            zip_name = f"{Path(dest).stem}_converted_{int(time.time())}.zip"
            zip_path = Path(UPLOAD_FOLDER) / zip_name
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                for p in files:
                    if p.exists() and p.is_file():
                        z.write(p, arcname=p.name)
            output_path = str(zip_path)
            warnings_list.append("Conversion multi-fichiers: résultat fourni en .zip")

        else:
            output_path = str(result)

        output_filename = os.path.basename(output_path)
        processing_time = time.time() - start_time
        return jsonify(
            success=True,
            message='Conversion terminée',
            output_file=output_filename,
            download_url=f"/download/{output_filename}",
            processing_time=f"{processing_time:.2f}s",
            warnings=warnings_list,
        )
    except Exception as e:
        print(f"Erreur conversion: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    # Important: les graphiques (PNG) sont affichés dans le navigateur via <img src=...>.
    # Pour éviter un téléchargement automatique, on sert les images en inline.
    # Pour le reste (CSV, JSON, etc.), le téléchargement est souvent plus logique.
    ext = os.path.splitext(str(filename))[1].lower()
    inline_exts = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}
    as_attachment = ext not in inline_exts
    if str(request.args.get('inline', '')).strip() == '1':
        as_attachment = False
    if str(request.args.get('download', '')).strip() == '1':
        as_attachment = True
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=as_attachment)


@app.route('/nettoyer-mvp', methods=['POST'])
def nettoyer_mvp():
    """Endpoint minimal attendu par la page Nettoyage (essaie de nettoyer le fichier uploadé en session)."""
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify(success=False, error='Aucun fichier uploadé en session'), 400
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        # Utiliser Cleaner pour nettoyer les données (remplace DataFrameCleaner)
        df = pd.read_csv(path)
        cleaner = Cleaner(
            remove_empty=True,
            remove_duplicates=True,
            trim_strings=True,
            normalize_spaces=True,
            fix_typo=True
        )
        df_clean = cleaner.clean(df)
        # save cleaned file
        cleaned_name = f"cleaned_{filename}"
        cleaned_path = os.path.join(UPLOAD_FOLDER, cleaned_name)
        df_clean.to_csv(cleaned_path, index=False)

        stats = {
            'lignes_avant': int(pd.read_csv(path).shape[0]) if filename.endswith('.csv') else None,
            'lignes_apres': int(df_clean.shape[0]),
            'lignes_supprimees': None,
            'operations_effectuees': 3
        }

        return jsonify(success=True, message='Nettoyage terminé', stats=stats, operations=['remove_duplicates','drop_empty','normalize_columns'], download_url=f'/uploads/{cleaned_name}')
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/convert', methods=['POST'])
def convert_file():
    payload = request.get_json() or {}
    target = payload.get('target_format') or payload.get('target')
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify(success=False, error='Aucun fichier uploadé en session'), 400
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        conv = FormatConverter(path)
        out = conv.convert(target)
        out_name = os.path.basename(out)
        return jsonify(success=True, message='Conversion terminée', download_url=f'/uploads/{out_name}')
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/transcribe', methods=['POST'])
def transcribe_shim():
    payload = request.get_json() or {}
    # language est conservé pour compat mais non utilisé par l'implémentation Whisper actuelle
    language = payload.get('language', 'fr')
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify(success=False, error='Aucun fichier uploadé en session'), 400
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        transcriber = _get_audio_transcriber()
        if transcriber is None:
            return jsonify(success=False, error='Module audio indisponible sur ce serveur'), 501
        out_txt = transcriber.transcribe(path, write_txt=True)
        # read transcription
        with open(out_txt, 'r', encoding='utf-8') as f:
            text = f.read()
        download_url = f'/uploads/{os.path.basename(out_txt)}'
        # move transcription file to uploads if not there
        if not os.path.exists(os.path.join(UPLOAD_FOLDER, os.path.basename(out_txt))):
            os.replace(out_txt, os.path.join(UPLOAD_FOLDER, os.path.basename(out_txt)))
        return jsonify(success=True, message='Transcription terminée', transcription=text, download_url=download_url)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/graph/generate', methods=['POST'])
def graph_generate():
    payload = request.get_json() or {}
    graph_type = payload.get('graph_type', 'histogram')
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify(success=False, error='Aucun fichier uploadé en session'), 400
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        df = pd.read_csv(path)
        import matplotlib.pyplot as plt
        out_img = os.path.join(UPLOAD_FOLDER, f"graph_{int(datetime.utcnow().timestamp())}.png")
        plt.figure(figsize=(8,4))
        if graph_type == 'histogram':
            num = df.select_dtypes(include='number')
            if num.shape[1] == 0:
                raise ValueError('Pas de colonnes numériques')
            num.iloc[:,0].hist(bins=30)
        elif graph_type == 'bar':
            df.iloc[:,0].value_counts().head(10).plot(kind='bar')
        else:
            # scatter: try first two numeric columns
            num = df.select_dtypes(include='number')
            if num.shape[1] < 2:
                raise ValueError('Pas assez de colonnes numériques')
            plt.scatter(num.iloc[:,0], num.iloc[:,1], s=10)

        plt.tight_layout()
        plt.savefig(out_img)
        plt.close()
        graph_url = f'/uploads/{os.path.basename(out_img)}'
        return jsonify(success=True, message='Graph généré', graph_url=graph_url)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/file-stats', methods=['GET'])
def api_file_stats():
    """Renvoie des informations basiques sur le fichier uploadé en session.
    Indique si c'est un fichier data (csv/xls/xlsx) et si au moins une colonne est entièrement numérique.
    """
    filename = session.get('uploaded_file') or request.args.get('file')
    if not filename:
        return jsonify(success=False, error='Aucun fichier en session'), 400
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify(success=False, error='Fichier introuvable'), 404
    ext = Path(path).suffix.lower()
    is_data = ext in ('.csv', '.xls', '.xlsx', '.ods')
    if not is_data:
        return jsonify(success=True, file_type=ext, is_data_file=False)

    try:
        if ext == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        row_count, col_count = df.shape
        numeric_cols = []
        for col in df.columns:
            col_data = df[col].dropna()
            if col_data.size == 0:
                continue
            converted = pd.to_numeric(col_data, errors='coerce')
            if converted.notna().all():
                numeric_cols.append(str(col))

        return jsonify(success=True, file_type='csv' if ext=='.csv' else 'excel', is_data_file=True,
                       has_numeric_column=len(numeric_cols)>0, numeric_columns=numeric_cols,
                       rows=int(row_count), cols=int(col_count))
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _load_dataframe_from_session():
    """Charge un DataFrame pandas depuis le fichier uploadé en session (CSV/Excel/JSON/TSV)."""
    filename = session.get('uploaded_file')
    if not filename:
        raise FileNotFoundError('Aucun fichier uploadé en session')
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError('Fichier introuvable: ' + path)
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in {'.tsv', '.txt'}:
        df = pd.read_csv(path, sep='\t')
    elif ext == '.json':
        try:
            df = pd.read_json(path)
        except Exception:
            df = pd.read_json(path, orient='records')
    else:
        df = pd.read_excel(path)
    return df, filename


def _load_dataframe_from_session_slot(slot: str = 'primary'):
    slot = (slot or 'primary').strip().lower()
    key = 'uploaded_file' if slot in {'primary', 'left', 'main', 'default'} else 'uploaded_file_right'
    filename = session.get(key)
    if not filename:
        raise FileNotFoundError('Aucun fichier uploadé en session pour slot=' + slot)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError('Fichier introuvable: ' + path)
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in {'.tsv', '.txt'}:
        df = pd.read_csv(path, sep='\t')
    elif ext == '.json':
        try:
            df = pd.read_json(path)
        except Exception:
            df = pd.read_json(path, orient='records')
    else:
        df = pd.read_excel(path)
    return df, filename


def _get_uploaded_file_path_from_session() -> str:
    filename = session.get('uploaded_file')
    if not filename:
        raise FileNotFoundError('Aucun fichier uploadé en session')
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError('Fichier introuvable: ' + path)
    return path


def _dashboard_preview_rows(df: pd.DataFrame, n: int = 20):
    head = df.head(max(1, min(int(n or 20), 200))).copy()
    # Index global pour édition ciblée. Si déjà présent (ex: preview filtrée), on le conserve.
    if "_row" not in head.columns:
        head.insert(0, "_row", list(range(0, int(head.shape[0]))))

    # Normalise NaN/NaT + sentinelles texte -> None pour l'UI
    head = head.where(pd.notna(head), None)
    sentinels = {'', 'null', 'none', 'nan', 'na', 'n/a', 'NULL', 'None', 'NaN', 'NA', 'N/A'}
    for c in head.columns:
        if str(c) == '_row':
            continue
        try:
            if head[c].dtype == object:
                head[c] = head[c].apply(lambda v: None if (isinstance(v, str) and v.strip() in sentinels) else v)
        except Exception:
            pass
    rows = []
    for row in head.to_dict(orient='records'):
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp,)):
                clean_row[str(k)] = v.isoformat()
            else:
                clean_row[str(k)] = v
        rows.append(clean_row)
    return rows


def _dashboard_apply_filters(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """Filtrage simple type Excel pour le dashboard.

    payload:
      - q: recherche texte (sur colonnes object/string)
      - filters: list[{column, op, value}]
    ops: equals, contains, regex, gt/gte/lt/lte, isnull, notnull
    """
    if not isinstance(payload, dict):
        return df

    work = df.copy()
    # Conserve un index global éditable même après filtrage
    if '_row' not in work.columns:
        work.insert(0, '_row', list(range(0, int(work.shape[0]))))
    q = payload.get('q')
    q = q.strip() if isinstance(q, str) else ''

    if q:
        text_cols = list(work.select_dtypes(include=['object', 'string']).columns)
        if text_cols:
            mask = None
            for c in text_cols:
                s = work[c].astype(str)
                m = s.str.contains(q, case=False, na=False)
                mask = m if mask is None else (mask | m)
            if mask is not None:
                work = work[mask]

    filters = payload.get('filters')
    if not isinstance(filters, list) or not filters:
        return work

    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get('column')
        op = str(f.get('op') or '').strip().lower()
        val = f.get('value')
        if not isinstance(col, str) or not col or col not in work.columns:
            continue
        if not op:
            continue

        s = work[col]

        # manquants (NA + tokens texte)
        if op in {'isnull', 'null'}:
            work = work[_dashboard_missing_mask(s)]
            continue
        if op in {'notnull', 'not_null'}:
            work = work[~_dashboard_missing_mask(s)]
            continue

        # regex / contains / equals
        if op in {'contains', 'regex', 'equals', 'eq', '='}:
            needle = '' if val is None else str(val)
            if op in {'equals', 'eq', '='}:
                work = work[s.astype(str).str.strip().str.lower() == needle.strip().lower()]
            elif op == 'contains':
                if needle.strip():
                    work = work[s.astype(str).str.contains(needle, case=False, na=False)]
            else:  # regex
                if needle.strip():
                    work = work[s.astype(str).str.contains(needle, regex=True, na=False)]
            continue

        # numeric compares
        s_num = pd.to_numeric(s, errors='coerce')
        try:
            x = float(val)
        except Exception:
            continue
        if op in {'gt', '>'}:
            work = work[s_num > x]
        elif op in {'gte', '>='}:
            work = work[s_num >= x]
        elif op in {'lt', '<'}:
            work = work[s_num < x]
        elif op in {'lte', '<='}:
            work = work[s_num <= x]

    return work


def _dashboard_write_back(df: pd.DataFrame, input_filename: str) -> tuple[str, str]:
    """Écrit le DataFrame dans uploads et retourne (out_name, out_path)."""
    base, ext = os.path.splitext(input_filename)
    ext = ext.lower().strip() or '.csv'
    unique = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    out_name = f"{base}_edit_{unique}{ext}"
    out_path = os.path.join(UPLOAD_FOLDER, out_name)

    if ext in {'.xlsx', '.xls'}:
        df.to_excel(out_path, index=False)
    elif ext == '.json':
        df.to_json(out_path, orient='records', force_ascii=False, date_format='iso')
    else:
        df.to_csv(out_path, index=False)
    return out_name, out_path


@app.route('/api/dashboard/preview', methods=['POST'])
def api_dashboard_preview():
    try:
        payload = request.get_json() or {}
        slot = (payload.get('slot') or 'primary').strip()
        limit = int(payload.get('limit', 50) or 50)
        limit = max(1, min(limit, 200))
        df, filename = _load_dataframe_from_session_slot(slot)
        filtered = _dashboard_apply_filters(df, payload)
        rows = _dashboard_preview_rows(filtered, n=limit)
        return jsonify(success=True, filename=filename, rows=rows, total_rows=int(filtered.shape[0]))
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/edit-cell', methods=['POST'])
def api_dashboard_edit_cell():
    try:
        payload = request.get_json() or {}
        row_i = payload.get('row')
        col = payload.get('column')
        value = payload.get('value')
        slot = (payload.get('slot') or 'primary').strip()

        if row_i is None or col is None:
            return jsonify(success=False, error='row and column required'), 400
        row_i = int(row_i)
        col = str(col)

        df, filename = _load_dataframe_from_session_slot(slot)
        if col not in df.columns:
            return jsonify(success=False, error='Colonne introuvable'), 400
        if row_i < 0 or row_i >= int(df.shape[0]):
            return jsonify(success=False, error='Index de ligne invalide'), 400

        # Coercition similaire à /api/lab/edit-cell
        if value is None:
            new_v = pd.NA
        else:
            svalue = str(value)
            if svalue.strip() == '':
                new_v = pd.NA
            else:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    new_v = pd.to_numeric(svalue, errors='coerce')
                elif pd.api.types.is_bool_dtype(dtype):
                    new_v = svalue.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
                else:
                    new_v = svalue

        df = df.copy()
        df.at[df.index[row_i], col] = new_v

        out_name, _out_path = _dashboard_write_back(df, filename)
        # remplace le fichier de la session (slot primary)
        if slot == 'primary':
            session['uploaded_file'] = out_name
        elif slot == 'right':
            session['uploaded_file_right'] = out_name
        session.modified = True

        return jsonify(success=True, row=row_i, column=col, output_file=out_name, download_url=f"/uploads/{out_name}")
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/rename-column', methods=['POST'])
def api_dashboard_rename_column():
    try:
        payload = request.get_json() or {}
        old = (payload.get('old') or '').strip()
        new = (payload.get('new') or '').strip()
        if not old or not new:
            return jsonify(success=False, error='old and new required'), 400
        df, filename = _load_dataframe_from_session_slot('primary')
        if old not in df.columns:
            return jsonify(success=False, error='Colonne introuvable'), 400
        df = df.copy()
        df = df.rename(columns={old: new})
        out_name, _ = _dashboard_write_back(df, filename)
        session['uploaded_file'] = out_name
        session.modified = True
        return jsonify(success=True, output_file=out_name, download_url=f"/uploads/{out_name}")
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/replace-regex', methods=['POST'])
def api_dashboard_replace_regex():
    try:
        payload = request.get_json() or {}
        column = (payload.get('column') or '').strip()
        pattern = payload.get('pattern')
        repl = payload.get('repl')
        regex = bool(payload.get('regex', True))
        if not column or pattern is None or repl is None:
            return jsonify(success=False, error='column, pattern, repl required'), 400
        df, filename = _load_dataframe_from_session_slot('primary')
        if column not in df.columns:
            return jsonify(success=False, error='Colonne introuvable'), 400
        df = df.copy()
        s = df[column].astype(str)
        df[column] = s.str.replace(str(pattern), str(repl), regex=regex)
        out_name, _ = _dashboard_write_back(df, filename)
        session['uploaded_file'] = out_name
        session.modified = True
        return jsonify(success=True, output_file=out_name, download_url=f"/uploads/{out_name}")
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/derive-column', methods=['POST'])
def api_dashboard_derive_column():
    try:
        payload = request.get_json() or {}
        kind = (payload.get('kind') or '').strip().lower()
        new_name = (payload.get('new_name') or '').strip()
        if not kind or not new_name:
            return jsonify(success=False, error='kind and new_name required'), 400

        df, filename = _load_dataframe_from_session_slot('primary')
        if new_name in df.columns:
            return jsonify(success=False, error='Le nom de colonne existe déjà'), 400
        df = df.copy()

        if kind == 'concat':
            a = payload.get('col_a')
            b = payload.get('col_b')
            sep = payload.get('sep', '')
            if not a or not b or a not in df.columns or b not in df.columns:
                return jsonify(success=False, error='col_a et col_b requis'), 400
            df[new_name] = df[a].astype(str).fillna('') + str(sep) + df[b].astype(str).fillna('')

        elif kind == 'numeric_op':
            a = payload.get('col_a')
            b = payload.get('col_b')
            op = (payload.get('op') or '').strip()
            if not a or not b or a not in df.columns or b not in df.columns:
                return jsonify(success=False, error='col_a et col_b requis'), 400
            sa = pd.to_numeric(df[a], errors='coerce')
            sb = pd.to_numeric(df[b], errors='coerce')
            if op == '+':
                df[new_name] = sa + sb
            elif op == '-':
                df[new_name] = sa - sb
            elif op == '*':
                df[new_name] = sa * sb
            elif op == '/':
                df[new_name] = sa / sb
            else:
                return jsonify(success=False, error='op invalide (+, -, *, /)'), 400

        elif kind == 'regex_extract':
            source = payload.get('source')
            pattern = payload.get('pattern')
            group = int(payload.get('group', 1) or 1)
            if not source or source not in df.columns or not pattern:
                return jsonify(success=False, error='source et pattern requis'), 400
            s = df[source].astype(str)
            extracted = s.str.extract(str(pattern), expand=True)
            idx = max(0, group - 1)
            if extracted.shape[1] <= idx:
                return jsonify(success=False, error='Groupe regex introuvable'), 400
            df[new_name] = extracted.iloc[:, idx]

        else:
            return jsonify(success=False, error='kind invalide'), 400

        out_name, _ = _dashboard_write_back(df, filename)
        session['uploaded_file'] = out_name
        session.modified = True

        return jsonify(success=True, output_file=out_name, download_url=f"/uploads/{out_name}")
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _dashboard_missing_by_col(df: pd.DataFrame):
    missing = {}
    for col in df.columns:
        s = df[col]
        m = _dashboard_missing_mask(s)
        missing[str(col)] = int(m.sum())
    return missing


_DASHBOARD_MISSING_TOKENS = {'', 'null', 'none', 'n/a', 'na'}


def _dashboard_missing_mask(series: pd.Series) -> pd.Series:
    """Masque des valeurs manquantes 'logiques' (NA + tokens texte)."""
    m = series.isna()
    try:
        if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            s_str = series.astype('string')
            # note: s_str peut contenir <NA>
            lowered = s_str.str.strip().str.lower()
            m = m | lowered.isin(list(_DASHBOARD_MISSING_TOKENS))
    except Exception:
        # fallback: ne pas casser le dashboard sur des dtypes exotiques
        pass
    return m


def _dashboard_normalize_missing_tokens(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Remplace les tokens de manquants (''/NULL/NA/...) par NA.

    Objectif: rendre les opérations (fill, filtres isnull, stats) cohérentes.
    """
    work = df.copy()
    cols = columns or [str(c) for c in work.columns]
    for c in cols:
        if c not in work.columns:
            continue
        s = work[c]
        try:
            if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
                s_str = s.astype('string')
                lowered = s_str.str.strip().str.lower()
                mask = lowered.isin(list(_DASHBOARD_MISSING_TOKENS))
                if mask.any():
                    s_str = s_str.mask(mask, pd.NA)
                    work[c] = s_str
        except Exception:
            continue
    return work


def _dashboard_suggestions(df: pd.DataFrame):
    suggestions = []
    try:
        dup = int(df.duplicated().sum())
    except Exception:
        dup = 0
    if dup > 0:
        suggestions.append({'recommended_action': 'remove_duplicates', 'reason': f'{dup} doublons détectés'})

    missing_by_col = _dashboard_missing_by_col(df)
    total_missing = sum(missing_by_col.values())
    if total_missing > 0:
        ratio = total_missing / max(int(df.shape[0] * df.shape[1]), 1)
        if ratio >= 0.05:
            suggestions.append({'recommended_action': 'fill_numeric_median', 'reason': 'Valeurs manquantes (remplissage)'})
        else:
            suggestions.append({'recommended_action': 'drop_rows_with_na', 'reason': 'Quelques valeurs manquantes (suppression lignes)'})

    try:
        empty_cols = [c for c in df.columns if df[c].isna().all() or (df[c].astype(str).str.strip() == '').all()]
    except Exception:
        empty_cols = []
    if empty_cols:
        suggestions.append({'recommended_action': 'remove_empty_cols', 'reason': f'{len(empty_cols)} colonne(s) vide(s)'})

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        suggestions.append({'recommended_action': 'remove_extra_spaces', 'reason': 'Colonnes texte détectées'})

    return suggestions


def _dashboard_normalize_column_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip('_')
    return s or "col"


@app.route('/api/dashboard/analyze', methods=['GET', 'POST'])
def api_dashboard_analyze():
    try:
        slot = request.args.get('slot')
        if not slot and request.is_json:
            slot = (request.get_json() or {}).get('slot')
        slot = slot or 'primary'

        df, filename = _load_dataframe_from_session_slot(slot)
        analysis = {
            'filename': filename,
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'columns': [str(c) for c in df.columns],
            'dtypes': {str(c): str(df[c].dtype) for c in df.columns},
            'missing_values': _dashboard_missing_by_col(df),
            'duplicates': int(df.duplicated().sum()),
            'preview': _dashboard_preview_rows(df, n=20),
            'suggestions': _dashboard_suggestions(df),
        }
        return jsonify(success=True, analysis=analysis)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/operations', methods=['POST'])
def api_dashboard_operations():
    try:
        payload = request.get_json() or {}
        ops = payload.get('operations') or []
        if not isinstance(ops, list) or not ops:
            return jsonify(success=False, error='operations must be a non-empty list'), 400

        # Scope global: all_rows | filtered_rows
        apply_to_default = str(payload.get('apply_to') or 'all_rows').strip().lower()
        if apply_to_default not in {'all_rows', 'filtered_rows'}:
            apply_to_default = 'all_rows'

        replace_session_file = bool(payload.get('replace_session_file', True))
        output_format = (payload.get('output_format') or 'csv').strip().lower()
        preview_rows = int(payload.get('preview_rows', 20) or 20)
        preview_rows = max(1, min(preview_rows, 200))

        df, filename = _load_dataframe_from_session_slot('primary')
        df = _dashboard_normalize_missing_tokens(df)
        results = []

        def _numeric_cols(frame: pd.DataFrame):
            return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

        def _parse_op_item(item):
            # legacy: string
            if isinstance(item, str):
                return str(item), {}
            if isinstance(item, dict):
                name = item.get('operation') or item.get('op') or item.get('name')
                return str(name or ''), dict(item)
            return '', {}

        # Cache index filtré (si demandé)
        filtered_index = None
        if apply_to_default == 'filtered_rows':
            try:
                filtered_index = _dashboard_apply_filters(df, payload).index
            except Exception:
                filtered_index = None

        for op_item in ops:
            try:
                op, op_params = _parse_op_item(op_item)
                op = (op or '').strip()
                if not op:
                    results.append({'operation': str(op_item), 'success': False, 'error': 'Opération invalide'})
                    continue

                op_apply_to = str(op_params.get('apply_to') or apply_to_default).strip().lower()
                if op_apply_to not in {'all_rows', 'filtered_rows'}:
                    op_apply_to = apply_to_default

                # colonnes ciblées (optionnel)
                cols_param = op_params.get('columns') or op_params.get('cols')
                if isinstance(cols_param, str) and cols_param:
                    target_cols = [cols_param]
                elif isinstance(cols_param, list):
                    target_cols = [str(c) for c in cols_param if c is not None and str(c).strip()]
                else:
                    target_cols = []

                def _get_row_index_for_op():
                    if op_apply_to == 'filtered_rows' and filtered_index is not None:
                        return filtered_index
                    return None

                row_index = _get_row_index_for_op()

                if op == 'remove_duplicates':
                    before = int(len(df))
                    df = df.drop_duplicates()
                    after = int(len(df))
                    results.append({'operation': op, 'success': True, 'message': f'{before - after} doublon(s) supprimé(s)'})

                elif op == 'remove_empty_rows':
                    before = int(len(df))
                    is_empty = df.isna().all(axis=1)
                    try:
                        is_empty = is_empty | df.astype(str).apply(lambda r: r.str.strip().eq('').all(), axis=1)
                    except Exception:
                        pass
                    df = df.loc[~is_empty].copy()
                    after = int(len(df))
                    results.append({'operation': op, 'success': True, 'message': f'{before - after} ligne(s) vide(s) supprimée(s)'})

                elif op == 'remove_empty_cols':
                    before = int(df.shape[1])
                    to_drop = []
                    for c in df.columns:
                        s = df[c]
                        if s.isna().all():
                            to_drop.append(c)
                            continue
                        if s.dtype == object:
                            try:
                                if s.astype(str).str.strip().eq('').all():
                                    to_drop.append(c)
                            except Exception:
                                pass
                    if to_drop:
                        df = df.drop(columns=to_drop)
                    after = int(df.shape[1])
                    results.append({'operation': op, 'success': True, 'message': f'{before - after} colonne(s) vide(s) supprimée(s)'})

                elif op == 'normalize_column_names':
                    before = list(df.columns)
                    new_cols = []
                    seen = {}
                    for c in before:
                        base = _dashboard_normalize_column_name(c)
                        idx = seen.get(base, 0)
                        seen[base] = idx + 1
                        new_cols.append(base if idx == 0 else f"{base}_{idx+1}")
                    df.columns = new_cols
                    results.append({'operation': op, 'success': True, 'message': 'Noms de colonnes normalisés'})

                elif op == 'fill_numeric_median':
                    cols = target_cols or _numeric_cols(df)
                    filled_cols = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        # calcul sur scope
                        s_all = pd.to_numeric(df[c], errors='coerce')
                        if row_index is not None:
                            s_scope = s_all.loc[row_index]
                        else:
                            s_scope = s_all
                        med = s_scope.median(skipna=True)
                        if pd.isna(med):
                            continue
                        if row_index is not None:
                            mask = s_all.loc[row_index].isna()
                            df.loc[row_index, c] = s_all.loc[row_index].mask(mask, med)
                        else:
                            df[c] = s_all.fillna(med)
                        filled_cols += 1
                    results.append({'operation': op, 'success': True, 'message': f'{filled_cols} colonne(s) numérique(s) remplie(s) (médiane)'})

                elif op == 'fill_numeric_mean':
                    cols = target_cols or _numeric_cols(df)
                    filled_cols = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        s_all = pd.to_numeric(df[c], errors='coerce')
                        if row_index is not None:
                            s_scope = s_all.loc[row_index]
                        else:
                            s_scope = s_all
                        mean = s_scope.mean(skipna=True)
                        if pd.isna(mean):
                            continue
                        if row_index is not None:
                            mask = s_all.loc[row_index].isna()
                            df.loc[row_index, c] = s_all.loc[row_index].mask(mask, mean)
                        else:
                            df[c] = s_all.fillna(mean)
                        filled_cols += 1
                    results.append({'operation': op, 'success': True, 'message': f'{filled_cols} colonne(s) numérique(s) remplie(s) (moyenne)'})

                elif op == 'fill_categorical_mode':
                    if target_cols:
                        cat_cols = [c for c in target_cols if c in df.columns]
                    else:
                        cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
                    filled = 0
                    for c in cat_cols:
                        s = df[c]
                        # normaliser tokens manquants sur la colonne
                        df = _dashboard_normalize_missing_tokens(df, columns=[str(c)])
                        s = df[c]
                        if row_index is not None:
                            s_scope = s.loc[row_index]
                        else:
                            s_scope = s
                        mode = None
                        try:
                            m = s_scope.mode(dropna=True)
                            if len(m):
                                mode = m.iloc[0]
                        except Exception:
                            mode = None
                        if mode is not None:
                            if row_index is not None:
                                mask = _dashboard_missing_mask(s.loc[row_index])
                                df.loc[row_index, c] = s.loc[row_index].mask(mask, mode)
                            else:
                                df[c] = s.mask(_dashboard_missing_mask(s), mode)
                            filled += 1
                    results.append({'operation': op, 'success': True, 'message': f'{filled} colonne(s) texte remplie(s) (mode)'})

                elif op == 'drop_rows_with_na':
                    before = int(len(df))
                    df = df.dropna()
                    after = int(len(df))
                    results.append({'operation': op, 'success': True, 'message': f'{before - after} ligne(s) supprimée(s) (NA)'})

                elif op == 'convert_to_numeric':
                    converted = 0
                    cols = target_cols or list(df.columns)
                    for c in cols:
                        if c not in df.columns:
                            continue
                        s = df[c]
                        # conversion forcée si l'utilisateur cible explicitement la colonne
                        forced = bool(target_cols)
                        if (s.dtype != object and not pd.api.types.is_string_dtype(s.dtype)) and not forced:
                            continue
                        s_str = s.astype('string')
                        s_str = s_str.str.replace(' ', '', regex=False)
                        s_str = s_str.str.replace(',', '.', regex=False)
                        num = pd.to_numeric(s_str, errors='coerce')
                        if forced or (num.notna().mean() >= 0.85 and num.notna().sum() >= 3):
                            df[c] = num
                            converted += 1
                    results.append({'operation': op, 'success': True, 'message': f'{converted} colonne(s) convertie(s) en numérique'})

                elif op == 'remove_numeric_outliers':
                    cols = target_cols or _numeric_cols(df)
                    if not cols:
                        results.append({'operation': op, 'success': True, 'message': 'Aucune colonne numérique'})
                    else:
                        clipped_total = 0
                        for c in cols:
                            if c not in df.columns:
                                continue
                            s = df[c]
                            q1 = s.quantile(0.25)
                            q3 = s.quantile(0.75)
                            iqr = q3 - q1
                            if pd.isna(iqr) or iqr == 0:
                                continue
                            lo = q1 - 1.5 * iqr
                            hi = q3 + 1.5 * iqr
                            before_out = int(((s < lo) | (s > hi)).sum())
                            if row_index is not None:
                                df.loc[row_index, c] = s.loc[row_index].clip(lower=lo, upper=hi)
                            else:
                                df[c] = s.clip(lower=lo, upper=hi)
                            clipped_total += before_out
                        results.append({'operation': op, 'success': True, 'message': f'{clipped_total} valeur(s) aberrante(s) clampée(s)'})

                elif op == 'normalize_numeric_values':
                    cols = _numeric_cols(df)
                    created = 0
                    for c in cols:
                        s = df[c]
                        mn = s.min(skipna=True)
                        mx = s.max(skipna=True)
                        if pd.isna(mn) or pd.isna(mx) or mx == mn:
                            continue
                        df[f"{c}__norm"] = (s - mn) / (mx - mn)
                        created += 1
                    results.append({'operation': op, 'success': True, 'message': f'{created} colonne(s) normalisée(s) ajoutée(s) (*__norm)'})

                elif op == 'clean_text_basic':
                    cols = target_cols or [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
                    changed = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        if row_index is not None:
                            s = df.loc[row_index, c].astype('string')
                            s = s.str.strip().str.replace(r"\s+", " ", regex=True)
                            df.loc[row_index, c] = s
                        else:
                            s = df[c].astype('string')
                            df[c] = s.str.strip().str.replace(r"\s+", " ", regex=True)
                        changed += 1
                    results.append({'operation': op, 'success': True, 'message': f'{changed} colonne(s) texte nettoyée(s)'})

                elif op == 'remove_html_tags':
                    cols = target_cols or [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
                    changed = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        if row_index is not None:
                            s = df.loc[row_index, c].astype('string')
                            df.loc[row_index, c] = s.str.replace(r"<[^>]+>", "", regex=True)
                        else:
                            s = df[c].astype('string')
                            df[c] = s.str.replace(r"<[^>]+>", "", regex=True)
                        changed += 1
                    results.append({'operation': op, 'success': True, 'message': f'Tags HTML supprimés ({changed} colonnes)'})

                elif op == 'standardize_text_case':
                    cols = target_cols or [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
                    changed = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        if row_index is not None:
                            df.loc[row_index, c] = df.loc[row_index, c].astype('string').str.lower()
                        else:
                            df[c] = df[c].astype('string').str.lower()
                        changed += 1
                    results.append({'operation': op, 'success': True, 'message': f'Casse standardisée ({changed} colonnes)'})

                elif op == 'remove_extra_spaces':
                    cols = target_cols or [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype)]
                    changed = 0
                    for c in cols:
                        if c not in df.columns:
                            continue
                        if row_index is not None:
                            s = df.loc[row_index, c].astype('string')
                            df.loc[row_index, c] = s.str.replace(r"\s+", " ", regex=True).str.strip()
                        else:
                            s = df[c].astype('string')
                            df[c] = s.str.replace(r"\s+", " ", regex=True).str.strip()
                        changed += 1
                    results.append({'operation': op, 'success': True, 'message': f'Espaces normalisés ({changed} colonnes)'})

                elif op == 'parse_dates':
                    converted = 0
                    cols = target_cols or list(df.columns)
                    for c in list(cols):
                        if c not in df.columns:
                            continue
                        s = df[c]
                        forced = bool(target_cols)
                        if (s.dtype != object and not pd.api.types.is_string_dtype(s.dtype)) and not forced:
                            continue
                        parsed = pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)
                        if forced or (parsed.notna().mean() >= 0.85 and parsed.notna().sum() >= 3):
                            df[c] = parsed
                            converted += 1
                    results.append({'operation': op, 'success': True, 'message': f'{converted} colonne(s) convertie(s) en date'})

                elif op == 'standardize_date_format':
                    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    created = 0
                    for c in dt_cols:
                        df[f"{c}__iso"] = df[c].dt.strftime('%Y-%m-%d')
                        created += 1
                    results.append({'operation': op, 'success': True, 'message': f'{created} colonne(s) format ISO ajoutée(s) (*__iso)'})

                elif op == 'extract_date_components':
                    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    created = 0
                    for c in dt_cols:
                        df[f"{c}__year"] = df[c].dt.year
                        df[f"{c}__month"] = df[c].dt.month
                        df[f"{c}__day"] = df[c].dt.day
                        created += 3
                    results.append({'operation': op, 'success': True, 'message': f'Composants de dates ajoutés ({created} colonne(s))'})

                else:
                    results.append({'operation': op, 'success': False, 'error': 'Opération inconnue'})

            except Exception as e:
                results.append({'operation': op_item if isinstance(op_item, str) else (op or 'operation'), 'success': False, 'error': str(e)})

        timestamp = int(datetime.utcnow().timestamp())
        base, _ext = os.path.splitext(filename)
        out_base = f"{base}_dashboard_{timestamp}"
        if output_format == 'excel':
            stored = f"{out_base}.xlsx"
            df.to_excel(os.path.join(UPLOAD_FOLDER, stored), index=False)
        elif output_format == 'json':
            stored = f"{out_base}.json"
            df.to_json(os.path.join(UPLOAD_FOLDER, stored), orient='records', indent=2)
        else:
            stored = f"{out_base}.csv"
            df.to_csv(os.path.join(UPLOAD_FOLDER, stored), index=False)

        if replace_session_file:
            session['uploaded_file'] = stored
            session.modified = True

        return jsonify(
            success=True,
            stats={
                'lignes_finales': int(df.shape[0]),
                'colonnes_finales': int(df.shape[1]),
                'operations_executees': int(len(ops)),
            },
            operations_results=results,
            preview=_dashboard_preview_rows(df, n=preview_rows),
            output_file=stored,
            download_url=f"/uploads/{stored}",
        )

    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/dashboard/merge', methods=['POST'])
def api_dashboard_merge():
    if DataManipulator is None or MergeSpec is None:
        return jsonify(success=False, error="Fonctionnalité indisponible (DataManipulator manquant)"), 501

    try:
        payload = request.get_json() or {}
        how = (payload.get('how') or 'left').strip().lower()
        left_on = payload.get('left_on')
        right_on = payload.get('right_on')
        on = payload.get('on')
        validate = payload.get('validate')
        indicator = bool(payload.get('indicator', False))
        suffixes = payload.get('suffixes') or ['_x', '_y']
        if not isinstance(suffixes, (list, tuple)) or len(suffixes) != 2:
            suffixes = ['_x', '_y']

        left_df, left_name = _load_dataframe_from_session_slot('primary')
        right_df, right_name = _load_dataframe_from_session_slot('right')

        spec_kwargs = {
            'how': how,
            'validate': validate,
            'indicator': indicator,
            'suffixes': (str(suffixes[0]), str(suffixes[1])),
        }
        if on:
            spec_kwargs['on'] = on
        else:
            if left_on:
                spec_kwargs['left_on'] = left_on
            if right_on:
                spec_kwargs['right_on'] = right_on

        spec = MergeSpec(**spec_kwargs)
        dm = DataManipulator()
        merged = dm.merge(left_df, right_df, spec)

        timestamp = int(datetime.utcnow().timestamp())
        base, _ext = os.path.splitext(left_name)
        out_name = f"{base}_merged_{timestamp}.csv"
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        merged.to_csv(out_path, index=False)

        session['uploaded_file'] = out_name
        session.modified = True

        return jsonify(
            success=True,
            message=f"Fusion terminée ({how})",
            stats={'lignes_finales': int(merged.shape[0]), 'colonnes_finales': int(merged.shape[1])},
            preview=_dashboard_preview_rows(merged, n=20),
            output_file=out_name,
            download_url=f"/uploads/{out_name}",
            left_file=left_name,
            right_file=right_name,
        )

    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except ValueError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/graph/catalog', methods=['GET'])
def api_graph_catalog():
    if GRAPH_SUPPORTED_TYPES is None:
        return jsonify(success=False, error="Fonctionnalité indisponible (modules.graph manquant)"), 501

    # Catalogue simple pour générer les formulaires côté UI
    catalog = {
        "graph_types": [
            {
                "graph_type": "correlation",
                "label": "Graphe de corrélation",
                "description": "Construit un graphe depuis les colonnes numériques (seuil de corrélation).",
                "schema": {
                    "threshold": {"type": "number", "default": 0.7, "min": 0.0, "max": 1.0},
                    "directed": {"type": "bool", "default": False},
                    "absolute": {"type": "bool", "default": True},
                    "min_degree": {"type": "int", "default": 1, "min": 0, "max": 1000},
                    "max_nodes": {"type": "int", "default": 200, "min": 10, "max": 2000},
                },
            },
            {
                "graph_type": "cosinus",
                "label": "Similarité cosinus (kNN par seuil)",
                "description": "Graphe basé sur similarité cosinus des lignes (numérique).",
                "schema": {
                    "threshold": {"type": "number", "default": 0.8, "min": 0.0, "max": 1.0},
                    "directed": {"type": "bool", "default": False},
                    "min_degree": {"type": "int", "default": 1, "min": 0, "max": 1000},
                    "max_nodes": {"type": "int", "default": 200, "min": 10, "max": 2000},
                },
            },
            {
                "graph_type": "knn",
                "label": "kNN",
                "description": "Construit un graphe k-nearest neighbors à partir des lignes (numérique).",
                "schema": {
                    "k": {"type": "int", "default": 5, "min": 1, "max": 100},
                    "directed": {"type": "bool", "default": False},
                    "min_degree": {"type": "int", "default": 1, "min": 0, "max": 1000},
                    "max_nodes": {"type": "int", "default": 300, "min": 10, "max": 5000},
                },
            },
            {
                "graph_type": "edgelist",
                "label": "Edge list (source/target)",
                "description": "Construit un graphe depuis des colonnes source/target (+ poids optionnel).",
                "schema": {
                    "source_col": {"type": "string", "default": "source"},
                    "target_col": {"type": "string", "default": "target"},
                    "weight_col": {"type": "string_optional", "default": "weight"},
                    "directed": {"type": "bool", "default": False},
                    "min_degree": {"type": "int", "default": 1, "min": 0, "max": 1000},
                    "max_nodes": {"type": "int", "default": 800, "min": 10, "max": 20000},
                },
            },
            {
                "graph_type": "cooccurrence",
                "label": "Co-occurrence (texte)",
                "description": "Graphe de co-occurrence à partir d'une colonne texte.",
                "schema": {
                    "text_column": {"type": "string_optional"},
                    "window": {"type": "int", "default": 2, "min": 1, "max": 10},
                    "min_count": {"type": "int", "default": 2, "min": 1, "max": 9999},
                    "lower": {"type": "bool", "default": True},
                    "min_degree": {"type": "int", "default": 1, "min": 0, "max": 1000},
                    "max_nodes": {"type": "int", "default": 250, "min": 10, "max": 5000},
                },
            },
        ],
        "supported": list(GRAPH_SUPPORTED_TYPES),
    }
    return jsonify(success=True, catalog=catalog)


@app.route('/api/graph/build', methods=['POST'])
def api_graph_build():
    if GRAPH_build_graph_from_files is None or GRAPH_build_graph_from_inputs is None:
        return jsonify(success=False, error="Fonctionnalité indisponible (modules.graph manquant)"), 501

    try:
        payload = request.get_json() or {}
        graph_type = str(payload.get('graph_type') or payload.get('type') or '').strip()
        if not graph_type:
            return jsonify(success=False, error='graph_type required'), 400

        options = payload.get('options') or {}
        if not isinstance(options, dict):
            return jsonify(success=False, error='options must be an object'), 400

        export_format = str(payload.get('export_format') or 'node_link').lower().strip()
        preview_layout = bool(payload.get('include_layout', True))
        min_degree = int(options.pop('min_degree', payload.get('min_degree', 1)) or 1)
        max_nodes = int(options.pop('max_nodes', payload.get('max_nodes', 200)) or 200)

        path = _get_uploaded_file_path_from_session()
        ext = Path(path).suffix.lower()

        graph_obj = None

        # Cooccurrence: possibilité de construire du texte depuis une colonne du dataset
        if graph_type == 'cooccurrence' and ext in {'.csv', '.tsv', '.xls', '.xlsx', '.ods'}:
            df, _fn = _load_dataframe_from_session()
            text_column = options.pop('text_column', None)
            if not text_column:
                # heuristique: première colonne texte
                text_cols = [c for c in df.columns if str(df[c].dtype) in {'object', 'string'}]
                text_column = str(text_cols[0]) if text_cols else None
            if not text_column or text_column not in df.columns:
                return jsonify(success=False, error='text_column introuvable (nécessaire pour cooccurrence)'), 400
            texts = [str(x) for x in df[text_column].dropna().astype(str).tolist()]
            graph_obj = GRAPH_build_graph_from_inputs(
                [GRAPH_InputItem(path=f"session:{_fn}:{text_column}", kind='txt', data='\n'.join(texts))],
                graph_type,
                **options,
            )

        # Excel/ODS: passer via DataFrame (types qui attendent un CSV)
        elif ext in {'.xls', '.xlsx', '.ods'} and graph_type in {'correlation', 'cosinus', 'knn', 'edgelist'}:
            df, _fn = _load_dataframe_from_session()
            graph_obj = GRAPH_build_graph_from_inputs(
                [GRAPH_InputItem(path=f"session:{_fn}", kind='csv', data=df)],
                graph_type,
                **options,
            )

        # Formats supportés directement par load_inputs
        else:
            graph_obj = GRAPH_build_graph_from_files(path, graph_type, **options)

        # Si le builder renvoie un dict (communities/doc_clusters), on exporte tel quel
        if isinstance(graph_obj, dict):
            data = GRAPH_export_graph_data(graph_obj, format='communities')
            out_name = f"graph_{graph_type}_{int(datetime.utcnow().timestamp())}.json"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            return jsonify(success=True, graph_type=graph_type, graph=data, metrics=None, layout=None, download_url=f"/uploads/{out_name}")

        # NetworkX graph
        import networkx as nx  # local import

        graph_nx: nx.Graph = graph_obj

        if GRAPH_filter_graph_by_degree is not None and min_degree > 0:
            graph_nx = GRAPH_filter_graph_by_degree(graph_nx, min_degree=min_degree)

        # Limiter taille par top degrés
        if max_nodes and graph_nx.number_of_nodes() > max_nodes:
            deg = sorted(graph_nx.degree(), key=lambda t: t[1], reverse=True)
            keep = set([n for n, _d in deg[:max_nodes]])
            graph_nx = graph_nx.subgraph(keep).copy()

        data = GRAPH_export_graph_data(graph_nx, format=export_format)

        metrics = GRAPH_compute_graph_metrics(graph_nx) if GRAPH_compute_graph_metrics is not None else None

        layout = None
        if preview_layout:
            try:
                pos = nx.spring_layout(graph_nx, seed=42)
                layout = {str(k): {'x': float(v[0]), 'y': float(v[1])} for k, v in pos.items()}
            except Exception:
                layout = None

        # Export téléchargeable (JSON node_link)
        out_name = f"graph_{graph_type}_{int(datetime.utcnow().timestamp())}.json"
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        try:
            txt = GRAPH_export_graph_text(graph_nx, format='node_link')
        except Exception:
            txt = json.dumps(data, ensure_ascii=False)
        Path(out_path).write_text(txt, encoding='utf-8')

        return jsonify(
            success=True,
            graph_type=graph_type,
            graph=data,
            metrics=metrics,
            layout=layout,
            download_url=f"/uploads/{out_name}",
        )

    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except ValueError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _dm_jsonify_preview(df: pd.DataFrame, *, n: int = 25):
    n = int(n or 25)
    n = max(1, min(n, 200))
    head = df.head(n).copy()
    head = head.where(pd.notna(head), None)

    def _coerce(v):
        if v is None:
            return None
        try:
            import numpy as _np

            if isinstance(v, (_np.generic,)):
                v = v.item()
        except Exception:
            pass
        try:
            if isinstance(v, (pd.Timestamp,)):
                return v.isoformat()
        except Exception:
            pass
        return v

    rows = []
    for row in head.to_dict(orient='records'):
        rows.append({str(k): _coerce(v) for k, v in row.items()})

    return {
        'columns': [str(c) for c in head.columns],
        'rows': rows,
        'preview_rows': int(len(rows)),
    }


@app.route('/api/data-manipulation/catalog', methods=['GET'])
def api_data_manipulation_catalog():
    if DataManipulator is None:
        return jsonify(success=False, error="Fonctionnalité indisponible (DataManipulator manquant)"), 501

    dm = DataManipulator()
    pattern_presets = []
    try:
        for name, rx in getattr(dm, 'feature_patterns', {}).items():
            if rx:
                pattern_presets.append({'key': name, 'label': name.replace('_', ' ').title(), 'pattern': rx})
    except Exception:
        pattern_presets = []

    catalog = {
        'operations': [
            {
                'op': 'rename_columns',
                'label': 'Renommer des colonnes',
                'description': 'Renomme une ou plusieurs colonnes via mapping.',
                'schema': {
                    'mapping': {'type': 'mapping', 'key': {'type': 'column'}, 'value': {'type': 'string'}},
                },
            },
            {
                'op': 'drop_columns',
                'label': 'Supprimer des colonnes',
                'description': 'Supprime une liste de colonnes.',
                'schema': {
                    'columns': {'type': 'columns', 'min': 1},
                    'errors': {'type': 'enum', 'choices': ['ignore', 'raise'], 'default': 'ignore'},
                },
            },
            {
                'op': 'replace_in_column',
                'label': 'Remplacer (regex) dans une colonne',
                'description': 'Effectue un remplacement regex (ou texte) dans une colonne.',
                'schema': {
                    'column': {'type': 'column'},
                    'pattern': {'type': 'string'},
                    'repl': {'type': 'string'},
                    'regex': {'type': 'bool', 'default': True},
                },
            },
            {
                'op': 'add_contact_and_id_features',
                'label': 'Détecter emails/tél/UUID/IP…',
                'description': 'Ajoute des flags has_<col>__email/uuid/ip/… sur colonnes texte.',
                'schema': {
                    'source_columns': {'type': 'columns_optional'},
                    'prefix': {'type': 'string', 'default': 'has_'},
                },
            },
            {
                'op': 'add_regex_flags',
                'label': 'Flags regex sur une colonne',
                'description': 'Ajoute des colonnes bool/int indiquant présence de motifs.',
                'schema': {
                    'source': {'type': 'column'},
                    'selected_presets': {'type': 'preset_multi', 'presets': pattern_presets},
                    'prefix': {'type': 'string', 'default': 'has_'},
                    'to_int': {'type': 'bool', 'default': True},
                },
            },
            {
                'op': 'add_unit_columns',
                'label': 'Extraire/convertir unités',
                'description': 'Extrait valeur+unité depuis une colonne texte et optionnellement convertit.',
                'schema': {
                    'source': {'type': 'column'},
                    'target_unit': {'type': 'string_optional'},
                    'exclude_currency': {'type': 'bool', 'default': True},
                },
            },
            {
                'op': 'apply_features',
                'label': 'Features déclaratives (avancé)',
                'description': 'Applique une liste de FeatureSpec (JSON).',
                'schema': {
                    'features': {'type': 'json'},
                },
            },
            {
                'op': 'clean',
                'label': 'Nettoyage (Cleaner)',
                'description': 'Applique le Cleaner si disponible (pipeline assistée).',
                'schema': {
                    'args': {'type': 'json_optional'},
                },
            },
        ]
    }
    return jsonify(success=True, catalog=catalog)


@app.route('/api/data-manipulation/run', methods=['POST'])
def api_data_manipulation_run():
    if DataManipulator is None:
        return jsonify(success=False, error="Fonctionnalité indisponible (DataManipulator manquant)"), 501

    try:
        payload = request.get_json() or {}
        preview_rows = int(payload.get('preview_rows', 25) or 25)
        output_format = (payload.get('output_format') or 'csv').lower().strip()
        replace_session_file = bool(payload.get('replace_session_file', False))

        df, filename = _load_dataframe_from_session()
        dm = DataManipulator()

        ops = payload.get('operations')
        if not ops:
            op = payload.get('op') or (payload.get('operation') or {}).get('op')
            args = payload.get('args') or (payload.get('operation') or {}).get('args') or {}
            if not op:
                return jsonify(success=False, error='Missing op or operations'), 400
            ops = [{'op': op, 'args': args}]
        if not isinstance(ops, list):
            return jsonify(success=False, error='operations must be a list'), 400

        for step in ops:
            op = (step or {}).get('op')
            args = (step or {}).get('args') or {}
            if not op:
                return jsonify(success=False, error='Invalid operation (missing op)'), 400

            if op == 'rename_columns':
                mapping = args.get('mapping')
                if not isinstance(mapping, dict) or not mapping:
                    return jsonify(success=False, error='rename_columns requires mapping dict'), 400
                df = dm.rename_columns(df, {str(k): str(v) for k, v in mapping.items()})

            elif op == 'drop_columns':
                cols = args.get('columns')
                if not isinstance(cols, list) or not cols:
                    return jsonify(success=False, error='drop_columns requires columns list'), 400
                df = dm.drop_columns(df, [str(c) for c in cols], errors=args.get('errors', 'ignore'))

            elif op == 'replace_in_column':
                column = args.get('column')
                pattern = args.get('pattern')
                repl = args.get('repl')
                if not column or pattern is None or repl is None:
                    return jsonify(success=False, error='replace_in_column requires column, pattern, repl'), 400
                df = dm.replace_in_column(
                    df,
                    column=str(column),
                    pattern=str(pattern),
                    repl=str(repl),
                    regex=bool(args.get('regex', True)),
                )

            elif op == 'add_contact_and_id_features':
                df = dm.add_contact_and_id_features(
                    df,
                    source_columns=args.get('source_columns'),
                    prefix=str(args.get('prefix', 'has_')),
                )

            elif op == 'add_regex_flags':
                source = args.get('source')
                if not source:
                    return jsonify(success=False, error='add_regex_flags requires source'), 400
                patterns = args.get('patterns')
                if patterns is None:
                    selected = args.get('selected_presets') or []
                    if not isinstance(selected, list):
                        selected = []
                    presets = getattr(dm, 'feature_patterns', {}) or {}
                    patterns = {k: presets.get(k) for k in selected if presets.get(k)}
                if not isinstance(patterns, dict) or not patterns:
                    return jsonify(success=False, error='add_regex_flags requires patterns (or selected_presets)'), 400
                df = dm.add_regex_flags(
                    df,
                    source=str(source),
                    patterns={str(k): str(v) for k, v in patterns.items() if v},
                    prefix=str(args.get('prefix', 'has_')),
                    to_int=bool(args.get('to_int', True)),
                )

            elif op == 'add_unit_columns':
                source = args.get('source')
                if not source:
                    return jsonify(success=False, error='add_unit_columns requires source'), 400
                df = dm.add_unit_columns(
                    df,
                    source=str(source),
                    target_unit=args.get('target_unit'),
                    exclude_currency=bool(args.get('exclude_currency', True)),
                )

            elif op == 'apply_features':
                feats = args.get('features')
                if feats is None:
                    feats = step.get('features')
                if not isinstance(feats, list) or not feats:
                    return jsonify(success=False, error='apply_features requires features list'), 400
                if FeatureSpec is None:
                    return jsonify(success=False, error='FeatureSpec unavailable'), 501
                specs = [FeatureSpec(**f) for f in feats]
                df = dm.apply_features(df, specs)

            elif op == 'clean':
                clean_args = args.get('args') if isinstance(args, dict) else {}
                clean_args = clean_args if isinstance(clean_args, dict) else {}
                df = dm.clean(df, **clean_args)

            else:
                return jsonify(success=False, error=f'Unsupported op: {op}'), 400

        base, _ext = os.path.splitext(filename)
        unique = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        out_name = None
        out_path = None

        if output_format == 'csv':
            out_name = f"dm_{base}_{unique}.csv"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            df.to_csv(out_path, index=False)
        elif output_format in {'excel', 'xlsx', 'xls'}:
            out_name = f"dm_{base}_{unique}.xlsx"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            df.to_excel(out_path, index=False)
        elif output_format == 'json':
            out_name = f"dm_{base}_{unique}.json"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            df.to_json(out_path, orient='records', force_ascii=False, date_format='iso')
        else:
            return jsonify(success=False, error='Unsupported output_format'), 400

        if replace_session_file and out_name:
            session['uploaded_file'] = out_name
            session.modified = True

        download_url = f"/uploads/{out_name}" if out_name else None

        return jsonify(
            success=True,
            input_filename=filename,
            output_filename=out_name,
            download_url=download_url,
            shape={'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
            preview=_dm_jsonify_preview(df, n=preview_rows),
            report=dm.get_report(),
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except KeyError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


# -----------------------------
 # API pour fonctionnalités numériques (modules/maths)
# -----------------------------
@app.route('/api/maths/correlation', methods=['POST'])
def api_maths_correlation():
    try:
        data = request.get_json() or request.form or {}
        col1 = data.get('col1')
        col2 = data.get('col2')
        method = data.get('method', 'pearson')
        if not col1 or not col2:
            return jsonify(success=False, error='col1 and col2 are required'), 400
        df, _ = _load_dataframe_from_session()
        analyzer = StatisticalAnalyzer(df)
        res = analyzer.correlation_test(col1, col2, method=method)
        return jsonify(success=True, result=res)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/normality', methods=['POST'])
def api_maths_normality():
    try:
        data = request.get_json() or request.form or {}
        col = data.get('col')
        method = data.get('method', 'shapiro')
        if not col:
            return jsonify(success=False, error='col is required'), 400
        df, _ = _load_dataframe_from_session()
        analyzer = StatisticalAnalyzer(df)
        res = analyzer.normality_test(col, method=method)
        return jsonify(success=True, result=res)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/pca', methods=['POST'])
@app.route('/api/sciences/pca', methods=['POST'])
def api_maths_pca():
    try:
        data = request.get_json() or request.form or {}
        n_components = int(data.get('n_components', 2))
        scale = bool(data.get('scale', True))
        df, filename = _load_dataframe_from_session()
        if MultivariateAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (MultivariateAnalyzer manquant)"), 501
        mv = MultivariateAnalyzer(df)
        res, fig = mv.run_pca(n_components=n_components, scale=scale, show=True)
        img_url = None
        if fig is not None:
            out_name = f"pca_{int(datetime.utcnow().timestamp())}.png"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            try:
                fig.savefig(out_path)
            except Exception:
                # fallback to plt
                import matplotlib.pyplot as plt
                plt.savefig(out_path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            img_url = f'/uploads/{out_name}'

        return jsonify(success=True, result={'explained_variance_ratio': res.get('explained_variance_ratio')}, plot_url=img_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/regression', methods=['POST'])
def api_maths_regression():
    try:
        data = request.get_json() or request.form or {}
        y = data.get('y')
        X = data.get('X') or data.get('features')
        if not y or not X:
            return jsonify(success=False, error='y (target) and X (features) are required'), 400
        # accept X as comma-separated string or list
        if isinstance(X, str):
            X = [c.strip() for c in X.split(',') if c.strip()]
        df, _ = _load_dataframe_from_session()
        if StatisticalAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (StatisticalAnalyzer manquant)"), 501
        analyzer = StatisticalAnalyzer(df)
        res = analyzer.linear_regression(y=y, X=X)
        return jsonify(success=True, result=res)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/ica', methods=['POST'])
@app.route('/api/sciences/ica', methods=['POST'])
def api_maths_ica():
    try:
        data = request.get_json() or request.form or {}
        n_components = int(data.get('n_components', 2))
        df, _ = _load_dataframe_from_session()
        if MultivariateAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (MultivariateAnalyzer manquant)"), 501
        mv = MultivariateAnalyzer(df)
        res, fig = mv.run_ica(n_components=n_components, show=True)
        plot_url = None
        if fig is not None:
            plot_url = _save_fig_to_uploads(fig, prefix='ica')
        return jsonify(success=True, result={'components_shape': None if res is None else (None)}, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/afc', methods=['POST'])
@app.route('/api/sciences/afc', methods=['POST'])
def api_maths_afc():
    try:
        data = request.get_json() or request.form or {}
        df, _ = _load_dataframe_from_session()
        if MultivariateAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (MultivariateAnalyzer manquant)"), 501
        mv = MultivariateAnalyzer(df)
        res, fig = mv.run_afc(show=True)
        plot_url = None
        if isinstance(fig, str):
            # some implementations may return message
            return jsonify(success=True, result=res)
        if fig is not None:
            plot_url = _save_fig_to_uploads(fig, prefix='afc')
        return jsonify(success=True, result={'afc': True}, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/multiple-correlation', methods=['POST'])
@app.route('/api/sciences/multiple-correlation', methods=['POST'])
def api_maths_multiple_correlation():
    try:
        data = request.get_json() or request.form or {}
        target = data.get('target')
        if not target:
            return jsonify(success=False, error='target required'), 400
        df, _ = _load_dataframe_from_session()
        if SpectralAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (SpectralAnalyzer manquant)"), 501
        # SpectralAnalyzer in this module also exposes multiple_correlation for df
        analyzer = SpectralAnalyzer(df)
        res = analyzer.multiple_correlation(target)
        return jsonify(success=True, result={'ranking': res})
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


# Session status endpoint (utilisé par lab.js)
@app.route('/api/session/status', methods=['GET'])
def api_session_status():
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify(success=True, has_file=False)
    path = os.path.join(UPLOAD_FOLDER, filename)
    exists = os.path.exists(path)
    ext = Path(path).suffix.lower() if exists else None
    is_data = ext in ('.csv', '.xls', '.xlsx', '.ods')
    return jsonify(success=True, has_file=exists, filename=filename, is_data_file=is_data)


@app.route('/api/lab/dataset-info', methods=['GET'])
def api_lab_dataset_info():
    try:
        df, filename = _load_dataframe_from_session()
        all_cols = [str(c) for c in df.columns]
        numeric_cols = [str(c) for c in df.select_dtypes(include='number').columns]
        datetime_cols = [str(c) for c in df.select_dtypes(include=['datetime', 'datetimetz']).columns]

        # Heuristique simple: text/categorical = objets + catégories + bool
        text_cols = [str(c) for c in df.select_dtypes(include=['object', 'string']).columns]
        categorical_cols = [str(c) for c in df.select_dtypes(include=['category', 'bool']).columns]
        # Ajouter aussi les colonnes texte dans 'categorical' pour les group_by (utilisable en pratique)
        categorical_cols = sorted(set(categorical_cols + text_cols))

        return jsonify(
            success=True,
            filename=filename,
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            columns=all_cols,
            numeric_columns=numeric_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            categorical_columns=categorical_cols,
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/summary', methods=['GET'])
def api_lab_summary():
    """Résumé descriptif du dataset en session.

    Objectif: donner immédiatement des infos utiles (manquants, uniques, stats de base)
    sans imposer de configuration ni afficher des corrélations.
    """
    try:
        df, filename = _load_dataframe_from_session()
        payload = _compute_basic_dataset_summary(df, filename=filename)
        return jsonify(success=True, **payload)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _compute_basic_dataset_summary(df: pd.DataFrame, filename: str | None = None) -> dict:
    """Calcule un résumé descriptif de base (manquants, uniques, stats numériques).

    Objectif: fournir une réponse stable pour l'UI (Lab/Dashboard).
    """
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    def _is_missing_value(series: pd.Series) -> pd.Series:
        m = series.isna()
        if series.dtype == object:
            s_str = series.astype(str)
            m = m | s_str.str.strip().isin(['', 'NULL', 'null', 'None', 'N/A', 'NA'])
        return m

    def _f(x):
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    columns_summary = []
    for col in df.columns:
        s = df[col]
        col_name = str(col)
        dtype = str(s.dtype)

        missing_mask = _is_missing_value(s)
        n_missing = int(missing_mask.sum())
        missing_pct = float((n_missing / n_rows) if n_rows else 0.0)

        s_non_missing = s[~missing_mask]
        n_unique = int(s_non_missing.nunique(dropna=True))
        unique_pct = float((n_unique / n_rows) if n_rows else 0.0)

        entry = {
            'column': col_name,
            'dtype': dtype,
            'missing': n_missing,
            'missing_pct': missing_pct,
            'unique': n_unique,
            'unique_pct': unique_pct,
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            entry.update({
                'mean': _f(desc.get('mean')),
                'std': _f(desc.get('std')),
                'min': _f(desc.get('min')),
                'p25': _f(desc.get('25%')),
                'median': _f(desc.get('50%')),
                'p75': _f(desc.get('75%')),
                'max': _f(desc.get('max')),
            })
        else:
            try:
                vc = s_non_missing.astype(str).value_counts().head(5)
                top_values = [{'value': str(k), 'count': int(v)} for k, v in vc.items()]
            except Exception:
                top_values = []
            entry['top_values'] = top_values

        try:
            examples = list(pd.unique(s_non_missing.astype(str)))[:5]
            entry['examples'] = [str(x) for x in examples]
        except Exception:
            entry['examples'] = []

        columns_summary.append(entry)

    columns_summary.sort(key=lambda x: (-x.get('missing', 0), x.get('column', '')))

    payload = {
        'filename': filename,
        'rows': n_rows,
        'cols': n_cols,
        'columns': columns_summary,
    }
    # Nettoyage: filename peut être None selon les usages
    if payload.get('filename') is None:
        payload.pop('filename', None)
    return payload


@app.route('/api/dashboard/summary', methods=['GET'])
def api_dashboard_summary():
    """Résumé descriptif du dataset pour le Dashboard (slot primary).

    Renvoie les mêmes stats de base que /api/lab/summary (moyenne, médiane, quantiles...).
    """
    try:
        df, filename = _load_dataframe_from_session_slot('primary')
        payload = _compute_basic_dataset_summary(df, filename=filename)
        return jsonify(success=True, **payload)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/preview', methods=['GET'])
def api_lab_preview():
    """Retourne un aperçu tabulaire du dataset en session.

    Utilisé pour le tri / l'affichage côté client (Lab Data Explorer).
    """
    try:
        limit = request.args.get('limit', default=200, type=int) or 200
        limit = max(1, min(int(limit), 1000))
        df, filename = _load_dataframe_from_session()

        head = df.iloc[:limit].copy()
        # Position globale (0..n-1) pour permettre l'édition ciblée sans ambiguïté.
        head.insert(0, "_row", list(range(0, int(head.shape[0]))))
        # Convertit NaN/NaT en None pour JSON.
        head = head.where(pd.notnull(head), None)

        columns = [str(c) for c in head.columns]
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        rows = head.to_dict(orient='records')

        return jsonify(
            success=True,
            filename=filename,
            columns=columns,
            dtypes=dtypes,
            rows=rows,
            total_rows=int(df.shape[0]),
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _apply_lab_query(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """Applique une requête simple (texte + filtres) sur un DataFrame.

    Format payload (tolérant):
      - q: str (recherche full-text sur colonnes texte)
      - filters: list[{column, op, value}]
    Ops supportés: equals, ne, contains, in, gt, gte, lt, lte, between, isnull, notnull
    """
    if not isinstance(payload, dict):
        return df

    work = df.reset_index(drop=True)

    q = payload.get('q')
    if isinstance(q, str):
        q = q.strip()
    else:
        q = ''

    # Full-text search on object/string columns
    if q:
        text_cols = list(work.select_dtypes(include=['object', 'string']).columns)
        if text_cols:
            mask = None
            for c in text_cols:
                s = work[c].astype(str)
                m = s.str.contains(q, case=False, na=False)
                mask = m if mask is None else (mask | m)
            if mask is not None:
                work = work[mask]

    filters = payload.get('filters')
    if not isinstance(filters, list):
        return work

    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get('column')
        op = str(f.get('op') or '').strip().lower()
        val = f.get('value')
        if not isinstance(col, str) or not col or col not in work.columns:
            continue
        if not op:
            continue

        series = work[col]

        if op in {'isnull', 'null'}:
            work = work[series.isna()]
            continue
        if op in {'notnull', 'not_null'}:
            work = work[~series.isna()]
            continue

        # Choose numeric vs string behavior
        is_num = pd.api.types.is_numeric_dtype(series)
        if is_num:
            s_num = pd.to_numeric(series, errors='coerce')
            if op in {'equals', 'eq', '='}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num == x]
            elif op in {'ne', '!='}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num != x]
            elif op in {'gt', '>'}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num > x]
            elif op in {'gte', '>='}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num >= x]
            elif op in {'lt', '<'}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num < x]
            elif op in {'lte', '<='}:
                try:
                    x = float(val)
                except Exception:
                    continue
                work = work[s_num <= x]
            elif op == 'between':
                lo = hi = None
                if isinstance(val, list) and len(val) >= 2:
                    lo, hi = val[0], val[1]
                elif isinstance(val, str) and ',' in val:
                    parts = [p.strip() for p in val.split(',') if p.strip()]
                    if len(parts) >= 2:
                        lo, hi = parts[0], parts[1]
                try:
                    lo_f = float(lo)
                    hi_f = float(hi)
                except Exception:
                    continue
                work = work[(s_num >= lo_f) & (s_num <= hi_f)]
            elif op == 'in':
                items = []
                if isinstance(val, list):
                    items = val
                elif isinstance(val, str):
                    items = [p.strip() for p in val.split(',') if p.strip()]
                parsed: list[float] = []
                for it in items:
                    try:
                        parsed.append(float(it))
                    except Exception:
                        continue
                if not parsed:
                    continue
                work = work[s_num.isin(parsed)]
            else:
                # contains for numeric doesn't make sense
                continue
        else:
            s_str = series.astype(str)
            if op in {'equals', 'eq', '='}:
                work = work[s_str.str.lower() == str(val).strip().lower()]
            elif op in {'ne', '!='}:
                work = work[s_str.str.lower() != str(val).strip().lower()]
            elif op == 'contains':
                needle = str(val).strip()
                if not needle:
                    continue
                work = work[s_str.str.contains(needle, case=False, na=False)]
            elif op == 'in':
                if isinstance(val, list):
                    items = [str(x).strip().lower() for x in val if str(x).strip()]
                else:
                    items = [p.strip().lower() for p in str(val).split(',') if p.strip()]
                if not items:
                    continue
                work = work[s_str.str.lower().isin(items)]
            else:
                continue

    return work


@app.route('/api/lab/query', methods=['POST'])
def api_lab_query():
    """Retourne un aperçu filtré du dataset (recherche + filtres)."""
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(success=False, error='payload must be an object'), 400

        limit = payload.get('limit', 200)
        try:
            limit = int(limit)
        except Exception:
            limit = 200
        limit = max(1, min(limit, 1000))

        df, filename = _load_dataframe_from_session()
        base = df.reset_index(drop=True)
        filtered = _apply_lab_query(base, payload)

        head = filtered.iloc[:limit].copy()
        head.insert(0, "_row", [int(i) for i in head.index.tolist()])
        head = head.where(pd.notnull(head), None)

        columns = [str(c) for c in head.columns]
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        rows = head.to_dict(orient='records')

        return jsonify(
            success=True,
            filename=filename,
            columns=columns,
            dtypes=dtypes,
            rows=rows,
            total_rows=int(base.shape[0]),
            filtered_rows=int(filtered.shape[0]),
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/export-query', methods=['POST'])
def api_lab_export_query():
    """Exporte le résultat d'une requête en CSV dans /uploads."""
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(success=False, error='payload must be an object'), 400

        df, filename = _load_dataframe_from_session()
        base = df.reset_index(drop=True)
        filtered = _apply_lab_query(base, payload)

        stem = Path(filename).stem
        out_name = f"filtered_{stem}_{int(datetime.utcnow().timestamp())}.csv"
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        filtered.to_csv(out_path, index=False)

        return jsonify(
            success=True,
            filename=out_name,
            rows=int(filtered.shape[0]),
            cols=int(filtered.shape[1]),
            download_url=f"/uploads/{out_name}",
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/transform', methods=['POST'])
def api_lab_transform():
    """Applique une transformation simple sur une colonne.

    Ne cherche pas à être un moteur d'expressions générique (sécurité/robustesse).
        Actions supportées:
            - log10, ln, abs, sqrt
            - multiply, add, power, round
            - minmax, zscore
            - upper, lower, strip
            - replace (find/replace littéral)
            - text_len, word_count, count_items
            - regex_extract_number
            - freq_encode, label_encode, one_hot_topk

    Body JSON:
      { action, column, mode: 'inplace'|'new', new_column?, value?, decimals?, find?, replace? }
    """
    try:
        payload = request.get_json(silent=True) or request.form or {}
        action = str(payload.get('action', '')).strip().lower()
        column = str(payload.get('column', '')).strip()
        columns = payload.get('columns', None)
        mode = str(payload.get('mode', 'new')).strip().lower()
        new_column = str(payload.get('new_column', '')).strip()

        if not action:
            return jsonify(success=False, error='action required'), 400

        df, filename = _load_dataframe_from_session()

        # --- Column deletion (cleaning) ---
        if action in {'drop_column', 'drop_columns', 'delete_column', 'delete_columns'}:
            # Accept either {column:"A"} or {columns:["A","B"]}
            cols_to_drop: list[str] = []
            if isinstance(columns, list):
                cols_to_drop = [str(c).strip() for c in columns if str(c).strip()]
            elif column:
                cols_to_drop = [column]

            # Normalize + unique
            cols_to_drop = list(dict.fromkeys(cols_to_drop))
            if not cols_to_drop:
                return jsonify(success=False, error='column(s) required'), 400

            missing = [c for c in cols_to_drop if c not in df.columns]
            if missing:
                return jsonify(success=False, error=f"Colonnes introuvables: {', '.join(missing)}"), 400

            remaining = [c for c in df.columns if c not in cols_to_drop]
            if len(remaining) < 1:
                return jsonify(success=False, error='Impossible de supprimer toutes les colonnes'), 400

            path = os.path.join(UPLOAD_FOLDER, filename)
            ext = Path(path).suffix.lower()
            if ext != '.csv':
                return jsonify(success=False, error='Transformations disponibles uniquement pour CSV dans Lab'), 400

            df = df.drop(columns=cols_to_drop)
            df.to_csv(path, index=False)
            session.modified = True

            return jsonify(
                success=True,
                filename=filename,
                changed=True,
                action='drop_column',
                dropped_columns=[str(c) for c in cols_to_drop],
                remaining_columns=[str(c) for c in df.columns],
            )

        # --- Standard column transforms ---
        if not column:
            return jsonify(success=False, error='column required'), 400
        if column not in df.columns:
            return jsonify(success=False, error=f"Colonne introuvable: {column}"), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        ext = Path(path).suffix.lower()
        if ext != '.csv':
            return jsonify(success=False, error='Transformations disponibles uniquement pour CSV dans Lab'), 400

        if mode not in {'inplace', 'new'}:
            mode = 'new'

        if mode == 'new':
            if not new_column:
                new_column = f"{column}_{action}".upper()
            # évite collision
            if new_column in df.columns:
                suffix = 2
                base = new_column
                while new_column in df.columns:
                    new_column = f"{base}_{suffix}"
                    suffix += 1
            target_col = new_column
        else:
            target_col = column

        s = df[column]

        # Numeric helpers
        def _to_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors='coerce')

        import numpy as np

        if action in {'log10', 'ln', 'sqrt', 'abs', 'multiply', 'add', 'power', 'round', 'minmax', 'zscore', 'linear'}:
            x = _to_numeric(s)

            if action in {'log10', 'ln'}:
                # log requiert x>0
                bad = x.dropna()
                if bad.empty:
                    return jsonify(success=False, error='Aucune valeur numérique valide dans la colonne'), 400
                if (bad <= 0).any():
                    return jsonify(success=False, error='Log impossible: valeurs <= 0 détectées'), 400
                y = np.log10(x) if action == 'log10' else np.log(x)

            elif action == 'sqrt':
                bad = x.dropna()
                if bad.empty:
                    return jsonify(success=False, error='Aucune valeur numérique valide dans la colonne'), 400
                if (bad < 0).any():
                    return jsonify(success=False, error='Racine impossible: valeurs < 0 détectées'), 400
                y = np.sqrt(x)

            elif action == 'abs':
                y = x.abs()

            elif action == 'multiply':
                value = payload.get('value', 1)
                try:
                    k = float(value)
                except Exception:
                    return jsonify(success=False, error='value must be numeric'), 400
                y = x * k

            elif action == 'add':
                value = payload.get('value', 0)
                try:
                    k = float(value)
                except Exception:
                    return jsonify(success=False, error='value must be numeric'), 400
                y = x + k

            elif action == 'linear':
                # x = a*y + b
                a_val = payload.get('a', None)
                b_val = payload.get('b', None)
                # Compat: si la UI envoie "value" sous forme "a,b"
                if (a_val is None and b_val is None) and payload.get('value') is not None:
                    raw = str(payload.get('value') or '').strip()
                    if raw:
                        parts = [p.strip() for p in raw.split(',')]
                        if len(parts) >= 1 and parts[0] != '':
                            a_val = parts[0]
                        if len(parts) >= 2 and parts[1] != '':
                            b_val = parts[1]

                try:
                    a = float(a_val) if a_val is not None and str(a_val).strip() != '' else 1.0
                except Exception:
                    return jsonify(success=False, error='a must be numeric'), 400
                try:
                    b = float(b_val) if b_val is not None and str(b_val).strip() != '' else 0.0
                except Exception:
                    return jsonify(success=False, error='b must be numeric'), 400

                y = (x * a) + b

            elif action == 'power':
                value = payload.get('value', 2)
                try:
                    p = float(value)
                except Exception:
                    return jsonify(success=False, error='value must be numeric'), 400
                y = np.power(x, p)

            else:  # round
                decimals = payload.get('decimals', 2)
                try:
                    d = int(decimals)
                except Exception:
                    d = 2
                y = x.round(d)

            if action == 'minmax':
                bad = x.dropna()
                if bad.empty:
                    return jsonify(success=False, error='Aucune valeur numérique valide dans la colonne'), 400
                vmin = float(bad.min())
                vmax = float(bad.max())
                denom = (vmax - vmin)
                if denom == 0:
                    y = x * 0.0
                else:
                    y = (x - vmin) / denom

            if action == 'zscore':
                bad = x.dropna()
                if bad.empty:
                    return jsonify(success=False, error='Aucune valeur numérique valide dans la colonne'), 400
                mu = float(bad.mean())
                sd = float(bad.std(ddof=0))
                if sd == 0:
                    y = x * 0.0
                else:
                    y = (x - mu) / sd

            df[target_col] = y

        elif action in {'upper', 'lower', 'strip'}:
            t = s.astype('string')
            if action == 'upper':
                df[target_col] = t.str.upper()
            elif action == 'lower':
                df[target_col] = t.str.lower()
            else:
                df[target_col] = t.str.strip()

        elif action == 'replace':
            find = payload.get('find', '')
            repl = payload.get('replace', '')
            if find is None or str(find) == '':
                return jsonify(success=False, error='find required'), 400
            t = s.astype('string')
            df[target_col] = t.str.replace(str(find), str(repl), regex=False)

        elif action in {
            'text_len',
            'word_count',
            'count_items',
            'regex_extract_number',
            'freq_encode',
            'label_encode',
            'one_hot_topk',
        }:
            t = s.astype('string')

            def _sanitize(name: str) -> str:
                safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
                safe = safe[:80] if safe else "col"
                return safe

            created_cols: list[str] = []

            if action == 'text_len':
                df[target_col] = t.str.len().astype('Int64')
                created_cols = [str(target_col)]

            elif action == 'word_count':
                df[target_col] = t.str.split().str.len().astype('Int64')
                created_cols = [str(target_col)]

            elif action == 'count_items':
                delim = payload.get('value', ',')
                delim = ',' if delim is None or str(delim) == '' else str(delim)

                def _count_items(v):
                    if v is None or pd.isna(v):
                        return pd.NA
                    items = [x.strip() for x in str(v).split(delim)]
                    items = [x for x in items if x]
                    return len(items)

                df[target_col] = t.apply(_count_items).astype('Int64')
                created_cols = [str(target_col)]

            elif action == 'regex_extract_number':
                pattern = payload.get('find', '')
                pattern = str(pattern).strip() if pattern is not None else ''
                if not pattern:
                    pattern = r"[-+]?\d*\.?\d+"
                try:
                    extracted = t.str.extract(pattern, expand=False)
                except Exception as e:
                    return jsonify(success=False, error=f"Regex invalide: {e}"), 400
                df[target_col] = pd.to_numeric(extracted, errors='coerce')
                created_cols = [str(target_col)]

            elif action == 'label_encode':
                non_na = t.dropna().astype(str)
                if non_na.empty:
                    return jsonify(success=False, error='Aucune valeur non-vide à encoder'), 400
                cats = sorted(non_na.unique().tolist())
                mapping = {c: i for i, c in enumerate(cats)}

                def _map_label(v):
                    if v is None or pd.isna(v):
                        return pd.NA
                    return mapping.get(str(v), pd.NA)

                df[target_col] = t.apply(_map_label).astype('Int64')
                created_cols = [str(target_col)]

            elif action == 'freq_encode':
                non_na = t.dropna().astype(str)
                if non_na.empty:
                    return jsonify(success=False, error='Aucune valeur non-vide à encoder'), 400
                counts = non_na.value_counts(dropna=True)
                mapping = counts.to_dict()

                def _map_freq(v):
                    if v is None or pd.isna(v):
                        return pd.NA
                    return int(mapping.get(str(v), 0))

                df[target_col] = t.apply(_map_freq).astype('Int64')
                created_cols = [str(target_col)]

            else:  # one_hot_topk
                if mode != 'new':
                    return jsonify(success=False, error='one_hot_topk nécessite "Nouvelle colonne" (crée plusieurs colonnes).'), 400

                raw_k = payload.get('value', 10)
                try:
                    k = int(raw_k)
                except Exception:
                    k = 10
                k = max(1, min(k, 30))

                prefix = str(target_col)
                if not prefix:
                    prefix = f"{column}_OH"

                non_na = t.dropna().astype(str)
                if non_na.empty:
                    return jsonify(success=False, error='Aucune valeur non-vide à encoder'), 400
                top = non_na.value_counts().head(k).index.tolist()

                def _unique_name(base: str) -> str:
                    base = _sanitize(base)
                    cand = base
                    suffix = 2
                    while cand in df.columns:
                        cand = f"{base}_{suffix}"
                        suffix += 1
                    return cand

                base_series = t.fillna('').astype(str)
                top_set = set(top)

                for cat in top:
                    colname = _unique_name(f"{prefix}__{cat}")
                    df[colname] = (base_series == str(cat)).astype(int)
                    created_cols.append(colname)

                other_col = _unique_name(f"{prefix}__OTHER")
                df[other_col] = ((base_series != '') & (~base_series.isin([str(x) for x in top_set]))).astype(int)
                created_cols.append(other_col)

        else:
            return jsonify(success=False, error=f"Action non supportée: {action}"), 400

        # Persist (Lab upload = CSV)
        df.to_csv(path, index=False)
        session.modified = True

        return jsonify(
            success=True,
            filename=filename,
            changed=True,
            target_column=str(target_col),
            created_columns=created_cols if 'created_cols' in locals() else [str(target_col)],
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/edit-cell', methods=['POST'])
def api_lab_edit_cell():
    """Édition type Excel d'une cellule (persistée dans le CSV).

    Body JSON:
      { row: int (position 0-based), column: str, value: any }
    """
    try:
        payload = request.get_json(silent=True) or request.form or {}
        row = payload.get('row')
        col = payload.get('column')
        value = payload.get('value')

        try:
            row_i = int(row)
        except Exception:
            return jsonify(success=False, error='row must be int'), 400
        if row_i < 0:
            return jsonify(success=False, error='row must be >= 0'), 400
        if not isinstance(col, str) or not col:
            return jsonify(success=False, error='column required'), 400

        df, filename = _load_dataframe_from_session()
        if col not in df.columns:
            return jsonify(success=False, error=f"Colonne introuvable: {col}"), 400
        if row_i >= int(df.shape[0]):
            return jsonify(success=False, error='row out of range'), 400

        path = os.path.join(UPLOAD_FOLDER, filename)
        ext = Path(path).suffix.lower()
        if ext != '.csv':
            return jsonify(success=False, error='Édition disponible uniquement pour CSV dans Lab'), 400

        # Excel-like: chaîne vide => NA
        if value is None:
            new_v = pd.NA
        else:
            svalue = str(value)
            if svalue.strip() == '':
                new_v = pd.NA
            else:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    new_v = pd.to_numeric(svalue, errors='coerce')
                elif pd.api.types.is_bool_dtype(dtype):
                    new_v = svalue.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
                else:
                    new_v = svalue

        df.at[df.index[row_i], col] = new_v

        df.to_csv(path, index=False)
        session.modified = True

        return jsonify(success=True, row=row_i, column=col)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


def _save_fig_to_uploads(fig, prefix='plot'):
    name = f"{prefix}_{int(datetime.utcnow().timestamp())}.png"
    path = os.path.join(UPLOAD_FOLDER, name)
    try:
        fig.savefig(path)
    except Exception:
        import matplotlib.pyplot as plt
        plt.savefig(path)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return f'/uploads/{name}'


def _polyfit_distribution_from_series(series: 'pd.Series', degree: int = 6, bins: int = 50):
    """Fit un polynôme sur une estimation simple de densité (histogramme).

    Retourne: (data, xs_dense, ys_dense, coeffs, formula_latex)
    """
    import numpy as np

    data = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    if data.size < max(10, degree + 2):
        raise ValueError("Pas assez de données numériques pour approximer la distribution")

    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    y = counts
    mask = np.isfinite(x_mid) & np.isfinite(y)
    x_mid = x_mid[mask]
    y = y[mask]
    if x_mid.size < 3:
        raise ValueError("Impossible d'estimer la distribution (données insuffisantes)")

    degree = max(1, int(degree))
    if x_mid.size <= degree:
        degree = max(1, int(x_mid.size - 1))

    with np.errstate(all='ignore'):
        coeffs = np.polyfit(x_mid, y, degree)
    poly = np.poly1d(coeffs)

    xs_dense = np.linspace(float(np.nanmin(x_mid)), float(np.nanmax(x_mid)), 300)
    ys_dense = poly(xs_dense)

    def _fmt(c: float) -> str:
        if not np.isfinite(c):
            return '0'
        s = f"{c:.6g}"
        return '0' if s in {'-0', '-0.0', '-0.00', '-0.000', '-0.0000'} else s

    terms = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        power = deg - i
        if not np.isfinite(c) or abs(c) < 1e-15:
            continue
        sign = '-' if c < 0 else '+'
        a = _fmt(abs(float(c)))
        if power == 0:
            term = f"{a}"
        elif power == 1:
            term = f"{a}x"
        else:
            term = f"{a}x^{power}"
        terms.append((sign, term))

    if not terms:
        formula = r"\\hat{f}(x)=0"
    else:
        first_sign, first_term = terms[0]
        expr = (first_term if first_sign == '+' else f"-{first_term}")
        for sgn, t in terms[1:]:
            expr += f" {sgn} {t}"
        formula = r"\\hat{f}(x)=" + expr

    return data, xs_dense, ys_dense, coeffs, formula


@app.route('/api/lab/scatter-plot', methods=['POST'])
def api_lab_scatter_plot():
    try:
        payload = request.get_json() or request.form or {}
        x_col = payload.get('x_column')
        y_col = payload.get('y_column')
        if not x_col or not y_col:
            return jsonify(success=False, error='x_column and y_column required'), 400
        df, _ = _load_dataframe_from_session()
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[y_col], errors='coerce')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(x, y, s=10, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{y_col} vs {x_col}')
        plot_url = _save_fig_to_uploads(fig, prefix='scatter')
        return jsonify(success=True, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/histogram', methods=['POST'])
def api_lab_histogram():
    try:
        payload = request.get_json() or request.form or {}
        col = payload.get('column')
        bins = int(payload.get('bins', 30))
        if not col:
            return jsonify(success=False, error='column required'), 400
        df, _ = _load_dataframe_from_session()
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(data, bins=bins, color='#4a90e2', alpha=0.8)
        ax.set_title(f'Histogramme: {col}')
        plot_url = _save_fig_to_uploads(fig, prefix='hist')
        return jsonify(success=True, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/boxplot', methods=['POST'])
def api_lab_boxplot():
    try:
        payload = request.get_json() or request.form or {}
        columns = payload.get('columns') or []
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(',') if c.strip()]
        if not columns:
            return jsonify(success=False, error='columns required'), 400
        df, _ = _load_dataframe_from_session()
        data = [pd.to_numeric(df[c], errors='coerce').dropna() for c in columns]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5))
        ax.boxplot(data, labels=columns)
        ax.set_title('Boxplot')
        plot_url = _save_fig_to_uploads(fig, prefix='boxplot')
        return jsonify(success=True, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/correlation-matrix', methods=['POST'])
def api_lab_correlation_matrix():
    try:
        payload = request.get_json() or request.form or {}
        columns = payload.get('columns')
        method = payload.get('method', 'pearson')
        df, _ = _load_dataframe_from_session()
        if columns:
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(',') if c.strip()]
            df_sub = df[columns]
        else:
            df_sub = df.select_dtypes(include='number')
        corr = df_sub.corr(method=method)
        # create heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        plot_url = _save_fig_to_uploads(fig, prefix='corr')
        return jsonify(success=True, matrix=corr.fillna(0).to_dict(), plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/linear-regression', methods=['POST'])
def api_lab_linear_regression():
    try:
        payload = request.get_json() or request.form or {}
        x_col = payload.get('x_column')
        y_col = payload.get('y_column')
        if not x_col or not y_col:
            return jsonify(success=False, error='x_column and y_column required'), 400
        df, _ = _load_dataframe_from_session()
        # use StatisticalAnalyzer for regression summary
        if StatisticalAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (StatisticalAnalyzer manquant)"), 501
        analyzer = StatisticalAnalyzer(df)
        res = analyzer.linear_regression(y=y_col, X=[x_col])
        # also produce scatter + line
        import numpy as np
        import matplotlib.pyplot as plt
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[y_col], errors='coerce')
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        if x_clean.size > 0:
            coef = np.polyfit(x_clean, y_clean, 1)
            poly1d = np.poly1d(coef)
            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter(x_clean, y_clean, s=8, alpha=0.6)
            xs = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(xs, poly1d(xs), color='red')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plot_url = _save_fig_to_uploads(fig, prefix='regression')
        else:
            plot_url = None
        return jsonify(success=True, result=res, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/polynomial-approximation', methods=['POST'])
def api_lab_polynomial_approximation():
    try:
        payload = request.get_json() or request.form or {}
        column = payload.get('column')
        degree = int(payload.get('degree', 6))
        bins = int(payload.get('bins', 50))
        df, _ = _load_dataframe_from_session()
        if not column or column not in df.columns:
            # choose first numeric column as fallback
            num = df.select_dtypes(include='number')
            if num.shape[1] == 0:
                return jsonify(success=False, error='Aucune colonne numérique pour approximation'), 400
            column = num.columns[0]

        # NOTE: l'implémentation vit dans modules.maths.fonctions.distribution
        # (moments_distribution ne définit pas DistributionApproximator dans ce repo).
        try:
            from modules.maths.fonctions.distribution import DistributionApproximator  # type: ignore
        except Exception:
            try:
                from modules.maths.fonctions.moments_distribution import DistributionApproximator  # type: ignore
            except Exception:
                DistributionApproximator = None

        # 1) Chemin "avancé" si disponible
        if DistributionApproximator is not None:
            try:
                approx = DistributionApproximator(df, column, degree=degree)
                approx._prepare_data()
                xs, ys, ys_fit = approx._fit_polynomial()
                formula = approx._latex_formula()

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(xs, ys, label='Empirique')
                ax.plot(xs, ys_fit, '--', label=f'Polynôme deg={degree}')
                ax.set_title(f'Approximation polynomiale ({column})')
                ax.legend()
                plot_url = _save_fig_to_uploads(fig, prefix='poly_approx')
                return jsonify(success=True, formula=formula, plot_url=plot_url)
            except Exception:
                # fallback si l'implémentation interne échoue
                pass

        # 2) Fallback robuste: polyfit sur histogramme de densité
        data, xs_dense, ys_dense, _coeffs, formula = _polyfit_distribution_from_series(
            df[column], degree=degree, bins=bins
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(data, bins=bins, density=True, color='#93c5fd', alpha=0.45, label='Histogramme (densité)')
        ax.plot(xs_dense, ys_dense, 'r--', linewidth=2, label=f'Polynôme deg={degree}')
        ax.set_title(f'Approximation polynomiale ({column})')
        ax.set_xlabel(column)
        ax.set_ylabel('Densité')
        ax.legend()
        plot_url = _save_fig_to_uploads(fig, prefix='poly_approx')

        return jsonify(success=True, formula=formula, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/distribution-analysis', methods=['POST'])
def api_lab_distribution_analysis():
    try:
        payload = request.get_json() or request.form or {}
        column = payload.get('column')
        degree = int(payload.get('degree', 6))
        bins = int(payload.get('bins', 50))
        df, _ = _load_dataframe_from_session()
        if not column or column not in df.columns:
            num = df.select_dtypes(include='number')
            if num.shape[1] == 0:
                return jsonify(success=False, error='Aucune colonne numérique'), 400
            column = num.columns[0]

        try:
            from modules.maths.fonctions.distribution import DistributionApproximator, DistributionComparator  # type: ignore
        except Exception:
            try:
                from modules.maths.fonctions.moments_distribution import DistributionApproximator, DistributionComparator  # type: ignore
            except Exception:
                DistributionApproximator = None
                DistributionComparator = None

        if DistributionApproximator is not None and DistributionComparator is not None:
            try:
                approx = DistributionApproximator(df, column, degree=degree)
                approx._prepare_data()
                xs, _ys, ys_fit = approx._fit_polynomial()
                formula = approx._latex_formula()

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7,4))
                data = pd.to_numeric(df[column], errors='coerce').dropna()
                ax.hist(data, bins=bins, density=True, color='#93c5fd', alpha=0.45, label='Histogramme (densité)')
                ax.plot(xs, ys_fit, 'r--', linewidth=2, label=f'Approx poly deg={degree}')
                ax.set_title(f'Distribution & approximation ({column})')
                ax.legend()
                plot_url = _save_fig_to_uploads(fig, prefix='dist_analysis')
                return jsonify(success=True, formula=formula, plot_url=plot_url)
            except Exception:
                pass

        # Fallback: polyfit sur histogramme de densité
        data, xs_dense, ys_dense, _coeffs, formula = _polyfit_distribution_from_series(
            df[column], degree=degree, bins=bins
        )

        import numpy as np
        summary = {
            'count': int(np.size(data)),
            'mean': float(np.nanmean(data)) if np.size(data) else None,
            'std': float(np.nanstd(data)) if np.size(data) else None,
            'min': float(np.nanmin(data)) if np.size(data) else None,
            'max': float(np.nanmax(data)) if np.size(data) else None,
        }

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(data, bins=bins, density=True, color='#93c5fd', alpha=0.45, label='Histogramme (densité)')
        ax.plot(xs_dense, ys_dense, 'r--', linewidth=2, label=f'Polynôme deg={degree}')
        ax.set_title(f'Distribution & approximation ({column})')
        ax.set_xlabel(column)
        ax.set_ylabel('Densité')
        ax.legend()
        plot_url = _save_fig_to_uploads(fig, prefix='dist_analysis')

        return jsonify(success=True, formula=formula, plot_url=plot_url, summary=summary)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/curve-fitting', methods=['POST'])
def api_lab_curve_fitting():
    try:
        payload = request.get_json() or request.form or {}
        x_column = payload.get('x_column')
        y_column = payload.get('y_column')
        method = payload.get('method', 'linear')
        df, _ = _load_dataframe_from_session()
        num = df.select_dtypes(include='number')
        if not x_column or x_column not in df.columns:
            if num.shape[1] < 2:
                return jsonify(success=False, error='Pas assez de colonnes numériques pour ajustement'), 400
            x_column = num.columns[0]
        if not y_column or y_column not in df.columns:
            y_column = num.columns[1] if num.shape[1] > 1 else num.columns[0]

        try:
            from modules.maths.fonctions.courbes_roc_pr import CurveFitter  # type: ignore
        except Exception:
            CurveFitter = None

        if CurveFitter is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (CurveFitter manquant)"), 501
        x = pd.to_numeric(df[x_column], errors='coerce').dropna()
        y = pd.to_numeric(df[y_column], errors='coerce').dropna()
        # align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        fitter = CurveFitter(x, y)
        fit_res = fitter.fit(method)

        # build plot
        func = fitter._get_model(method)
        import numpy as np
        import matplotlib.pyplot as plt
        x_vals = np.linspace(min(x), max(x), 200)
        y_fit = func(x_vals, *fit_res['params'])
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.plot(x_vals, y_fit, color='red')
        ax.set_title(f'Ajustement {method} : {y_column} ~ {x_column}')
        plot_url = _save_fig_to_uploads(fig, prefix='curvefit')

        return jsonify(success=True, params=fit_res['params'].tolist() if hasattr(fit_res['params'], 'tolist') else list(fit_res['params']), plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/distribution-compare', methods=['POST'])
def api_lab_distribution_compare():
    try:
        payload = request.get_json() or request.form or {}
        column = payload.get('column')
        compare_to = payload.get('compare_to')  # other column name or 'normal'
        df, _ = _load_dataframe_from_session()
        if not column or column not in df.columns:
            num = df.select_dtypes(include='number')
            if num.shape[1] == 0:
                return jsonify(success=False, error='Aucune colonne numérique'), 400
            column = num.columns[0]

        try:
            from modules.maths.fonctions.distribution import DistributionComparator  # type: ignore
        except Exception:
            try:
                from modules.maths.fonctions.moments_distribution import DistributionComparator  # type: ignore
            except Exception:
                DistributionComparator = None

        if DistributionComparator is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (DistributionComparator manquant)"), 501
        comp = DistributionComparator(df, column)
        comp.run()
        result = {}
        if compare_to and compare_to in df.columns:
            result = comp.compare_with_column(compare_to)
        else:
            # compare to normal fitted on data
            mu = float(df[column].dropna().mean())
            sigma = float(df[column].dropna().std())
            theoretical = comp.generate_theoretical('normal', loc=mu, scale=sigma)
            result = {'note': 'comparison with fitted normal', 'theoretical_sample': theoretical.tolist() if hasattr(theoretical, 'tolist') else list(theoretical)}

        # produce histogram + KDE plot for visual
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(pd.to_numeric(df[column], errors='coerce').dropna(), kde=True, ax=ax, color='lightgreen')
        ax.set_title(f'Distribution comparaison: {column}')
        plot_url = _save_fig_to_uploads(fig, prefix='dist_compare')

        return jsonify(success=True, result=result, plot_url=plot_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/descriptive-stats', methods=['POST'])
def api_lab_descriptive_stats():
    # Alias attendu par l'UI Lab vers l'endpoint stats existant (sciences/maths)
    return api_maths_statistical_describe()


@app.route('/api/lab/hypothesis-test', methods=['POST'])
def api_lab_hypothesis_test():
    try:
        payload = request.get_json() or request.form or {}
        test_type = (payload.get('test_type') or 'correlation').strip().lower()
        df, _ = _load_dataframe_from_session()

        if StatisticalAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (StatisticalAnalyzer manquant)"), 501

        sa = StatisticalAnalyzer(df)

        if test_type == 'correlation':
            col1 = payload.get('var1') or payload.get('col1')
            col2 = payload.get('var2') or payload.get('col2')
            method = payload.get('method', 'pearson')
            if not col1 or not col2:
                return jsonify(success=False, error='var1 et var2 requis pour correlation'), 400
            res = sa.correlation_test(str(col1), str(col2), method=method)
        elif test_type == 'normality':
            col = payload.get('var1') or payload.get('column')
            method = payload.get('method', 'shapiro')
            if not col:
                return jsonify(success=False, error='var1 requis pour normality'), 400
            res = sa.normality_test(str(col), method=method)
        elif test_type in {'ttest', 't-test', 'ttest_independent'}:
            col = payload.get('var1') or payload.get('column')
            group_col = payload.get('group_col') or payload.get('var2')
            if not col or not group_col:
                return jsonify(success=False, error='var1 (col) et group_col requis pour ttest'), 400
            res = sa.ttest_independent(str(col), str(group_col))
        elif test_type == 'anova':
            col = payload.get('var1') or payload.get('column')
            group_col = payload.get('group_col') or payload.get('var2')
            if not col or not group_col:
                return jsonify(success=False, error='var1 (col) et group_col requis pour anova'), 400
            res = sa.anova_oneway(str(col), str(group_col))
        elif test_type in {'chi2', 'chi-square', 'chisq'}:
            col1 = payload.get('var1') or payload.get('col1')
            col2 = payload.get('var2') or payload.get('col2')
            if not col1 or not col2:
                return jsonify(success=False, error='var1 et var2 requis pour chi2'), 400
            res = sa.chi2_test(str(col1), str(col2))
        else:
            return jsonify(success=False, error=f"test_type non supporté: {test_type}"), 400

        return jsonify(success=True, result=res)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/outlier-detection', methods=['POST'])
def api_lab_outlier_detection():
    try:
        payload = request.get_json() or request.form or {}
        method = (payload.get('method') or 'iqr').strip().lower()
        df, _ = _load_dataframe_from_session()
        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] == 0:
            return jsonify(success=False, error='Aucune colonne numérique pour détecter des outliers'), 400

        warning = None
        by_col = {}
        outlier_mask = None

        if method == 'zscore':
            import numpy as np
            threshold = float(payload.get('threshold', 3.0))
            z = (numeric - numeric.mean(numeric_only=True)) / numeric.std(numeric_only=True).replace(0, np.nan)
            outlier_mask = (z.abs() > threshold).any(axis=1)
            for c in numeric.columns:
                by_col[str(c)] = int((z[c].abs() > threshold).sum(skipna=True))
        elif method == 'iqr':
            import numpy as np
            k = float(payload.get('k', 1.5))
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            outlier_mask = ((numeric < lower) | (numeric > upper)).any(axis=1)
            for c in numeric.columns:
                by_col[str(c)] = int(((numeric[c] < lower[c]) | (numeric[c] > upper[c])).sum(skipna=True))
        elif method == 'isolation_forest':
            try:
                import numpy as np
                from sklearn.ensemble import IsolationForest  # type: ignore

                X = numeric.copy()
                for c in X.columns:
                    X[c] = X[c].fillna(X[c].median())
                clf = IsolationForest(
                    n_estimators=int(payload.get('n_estimators', 200)),
                    contamination=float(payload.get('contamination', 0.05)),
                    random_state=42,
                )
                preds = clf.fit_predict(X)
                outlier_mask = (preds == -1)
                # approx per-column contribution not available; provide IQR counts as complement
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = (q3 - q1).replace(0, np.nan)
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                for c in numeric.columns:
                    by_col[str(c)] = int(((numeric[c] < lower[c]) | (numeric[c] > upper[c])).sum(skipna=True))
            except Exception:
                warning = "IsolationForest indisponible; fallback sur IQR"
                method = 'iqr'
                import numpy as np
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = (q3 - q1).replace(0, np.nan)
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = ((numeric < lower) | (numeric > upper)).any(axis=1)
                for c in numeric.columns:
                    by_col[str(c)] = int(((numeric[c] < lower[c]) | (numeric[c] > upper[c])).sum(skipna=True))
        else:
            return jsonify(success=False, error=f"Méthode non supportée: {method}"), 400

        total_outliers = int(outlier_mask.sum()) if hasattr(outlier_mask, 'sum') else int(sum(outlier_mask))
        outlier_rows = list(df.index[outlier_mask][:50])

        plot_url = None
        try:
            import matplotlib.pyplot as plt
            cols = list(by_col.keys())
            vals = [by_col[c] for c in cols]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(cols, vals, color='#ef4444', alpha=0.85)
            ax.set_title('Outliers par colonne (approx)')
            ax.set_ylabel('Nombre')
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            plot_url = _save_fig_to_uploads(fig, prefix='outliers')
        except Exception:
            plot_url = None

        return jsonify(
            success=True,
            result={
                'method': method,
                'warning': warning,
                'total_rows': int(df.shape[0]),
                'total_outliers': total_outliers,
                'outlier_rows_sample': outlier_rows,
                'by_column': by_col,
            },
            plot_url=plot_url,
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/time-series', methods=['POST'])
def api_lab_time_series():
    try:
        df, _ = _load_dataframe_from_session()
        payload = request.get_json() or request.form or {}

        # 1) détecter une colonne date/temps
        date_col = None
        datetime_cols = list(df.select_dtypes(include=['datetime', 'datetimetz']).columns)
        if datetime_cols:
            date_col = datetime_cols[0]
        else:
            # tenter de parser des colonnes object
            best = (None, 0.0)
            for c in df.select_dtypes(include=['object', 'string']).columns:
                parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                frac = float(parsed.notna().mean()) if len(parsed) else 0.0
                if frac > best[1]:
                    best = (c, frac)
            if best[0] is not None and best[1] >= 0.5:
                date_col = best[0]

        numeric_cols = list(df.select_dtypes(include='number').columns)
        if not date_col or len(numeric_cols) == 0:
            return jsonify(success=False, error="Impossible d'auto-détecter une série temporelle (date + valeur numérique)"), 400

        value_col = numeric_cols[0]
        ts = df[[date_col, value_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce', infer_datetime_format=True)
        ts[value_col] = pd.to_numeric(ts[value_col], errors='coerce')
        ts = ts.dropna(subset=[date_col, value_col]).sort_values(date_col)
        if ts.shape[0] < 5:
            return jsonify(success=False, error='Pas assez de points temporels exploitables'), 400

        # inférer une fréquence si possible
        inferred = None
        try:
            inferred = pd.infer_freq(ts[date_col])
        except Exception:
            inferred = None

        # construire une série et un lissage
        series = ts.set_index(date_col)[value_col]
        # éviter les index non uniques
        series = series.groupby(level=0).mean().sort_index()
        rolling_window = int(payload.get('rolling_window', 7))
        smooth = series.rolling(window=max(2, rolling_window), min_periods=2).mean()

        plot_url = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(series.index, series.values, label='Valeur', linewidth=1.2)
            ax.plot(smooth.index, smooth.values, label=f'Moyenne mobile ({rolling_window})', linewidth=2.0)
            ax.set_title(f'Série temporelle: {value_col} vs {date_col}')
            ax.set_xlabel(str(date_col))
            ax.set_ylabel(str(value_col))
            ax.legend()
            fig.tight_layout()
            plot_url = _save_fig_to_uploads(fig, prefix='timeseries')
        except Exception:
            plot_url = None

        preview = ts.head(20).to_dict(orient='records')
        return jsonify(
            success=True,
            result={
                'date_column': str(date_col),
                'value_column': str(value_col),
                'n_points': int(ts.shape[0]),
                'inferred_frequency': inferred,
                'start': ts[date_col].min().isoformat() if hasattr(ts[date_col].min(), 'isoformat') else str(ts[date_col].min()),
                'end': ts[date_col].max().isoformat() if hasattr(ts[date_col].max(), 'isoformat') else str(ts[date_col].max()),
                'preview': preview,
            },
            plot_url=plot_url,
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/lab/classification', methods=['POST'])
def api_lab_classification():
    try:
        df, _ = _load_dataframe_from_session()
        payload = request.get_json() or request.form or {}

        # Auto-détection d'une cible binaire
        target = payload.get('target')
        if target and target in df.columns:
            target_col = target
        else:
            target_col = None
            for c in reversed(list(df.columns)):
                nunique = df[c].dropna().nunique()
                if nunique == 2:
                    target_col = c
                    break
        if target_col is None:
            return jsonify(success=False, error='Aucune colonne binaire détectée pour classification (2 classes requises)'), 400

        # Construire y binaire (0/1)
        y_raw = df[target_col]
        if set(y_raw.dropna().unique()) <= {0, 1}:
            y = y_raw.astype('float').astype('Int64')
            mapping = None
        else:
            classes, y_codes = pd.factorize(y_raw)
            if len(classes) != 2:
                return jsonify(success=False, error='La cible doit avoir exactement 2 classes'), 400
            y = pd.Series(y_codes, index=df.index)
            mapping = {str(classes[0]): 0, str(classes[1]): 1}

        # Features numériques
        X = df.select_dtypes(include='number').copy()
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        if X.shape[1] == 0:
            return jsonify(success=False, error='Aucune feature numérique disponible pour entraîner un modèle'), 400

        # Nettoyage lignes NaN
        work = X.copy()
        work['_y'] = y
        work = work.dropna()
        y_clean = work.pop('_y').astype(int)
        X_clean = work

        try:
            import statsmodels.api as sm
        except Exception:
            return jsonify(success=False, error='statsmodels indisponible pour la classification'), 501

        X_sm = sm.add_constant(X_clean, has_constant='add')
        model = sm.Logit(y_clean, X_sm).fit(disp=False)
        proba = model.predict(X_sm)
        pred = (proba >= 0.5).astype(int)
        accuracy = float((pred == y_clean).mean())

        coef = {k: float(v) for k, v in model.params.to_dict().items()}
        return jsonify(
            success=True,
            result={
                'target': str(target_col),
                'mapping': mapping,
                'features': [str(c) for c in X_clean.columns],
                'n_rows_used': int(X_clean.shape[0]),
                'train_accuracy': accuracy,
                'coefficients': coef,
                'summary': model.summary().as_text(),
            },
        )
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


# ============================
# Plot Studio (modules/plots)
# ============================


@app.route('/api/plots/catalog', methods=['GET'])
def api_plots_catalog():
    try:
        if PLOTS_get_plot_catalog is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (modules.plots/plotly manquants)"), 501
        return jsonify(success=True, catalog=PLOTS_get_plot_catalog())
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/plots/render', methods=['POST'])
def api_plots_render():
    try:
        if PLOTS_make_figure_json is None or PLOTS_get_plot_catalog is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (modules.plots/plotly manquants)"), 501

        spec = request.get_json() or request.form or {}
        if not isinstance(spec, dict):
            return jsonify(success=False, error='spec must be a JSON object'), 400

        plot_type = spec.get('plot_type')
        if not plot_type or not isinstance(plot_type, str):
            return jsonify(success=False, error='plot_type is required'), 400

        df, _ = _load_dataframe_from_session()

        # Optional: allow filtering the dataset before plotting.
        query = spec.get('query') if isinstance(spec, dict) else None
        if isinstance(query, dict):
            df = _apply_lab_query(df, query)
            spec = dict(spec)
            spec.pop('query', None)

        fig_json = PLOTS_make_figure_json(df, spec)

        # Plotly figures can contain numpy arrays; use Plotly's JSON encoder.
        try:
            from plotly.utils import PlotlyJSONEncoder
            payload = json.dumps(fig_json, cls=PlotlyJSONEncoder)
        except Exception:
            payload = json.dumps(fig_json)

        return jsonify(success=True, plotly_json=payload)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/multivariate', methods=['POST'])
def api_maths_multivariate():
    try:
        payload = request.get_json() or request.form or {}
        action = payload.get('action', 'report')  # 'report'|'pca'|'ica'|'afc'
        df, _ = _load_dataframe_from_session()
        if MultivariateAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (MultivariateAnalyzer manquant)"), 501
        mv = MultivariateAnalyzer(df)
        if action == 'pca':
            res, fig = mv.run_pca(n_components=int(payload.get('n_components', 2)), scale=True, show=True)
            plot_url = _save_fig_to_uploads(fig, prefix='multivar_pca') if fig is not None else None
            return jsonify(success=True, result={'explained_variance_ratio': res.get('explained_variance_ratio')}, plot_url=plot_url)
        elif action == 'ica':
            res, fig = mv.run_ica(n_components=int(payload.get('n_components', 2)), show=True)
            plot_url = _save_fig_to_uploads(fig, prefix='multivar_ica') if fig is not None else None
            return jsonify(success=True, result={'ica': True}, plot_url=plot_url)
        elif action == 'afc':
            res, fig = mv.run_afc(show=True)
            plot_url = _save_fig_to_uploads(fig, prefix='multivar_afc') if fig is not None else None
            return jsonify(success=True, result={'afc': True}, plot_url=plot_url)
        else:
            report = mv.generate_report()
            return jsonify(success=True, report=report)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/spectral-summary', methods=['POST'])
def api_maths_spectral_summary():
    try:
        # This endpoint will compute correlation matrix and hierarchical clustering via SpectralAnalyzer utilities
        df, _ = _load_dataframe_from_session()
        if SpectralAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (SpectralAnalyzer manquant)"), 501
        analyzer = SpectralAnalyzer(df)
        corr = analyzer.correlation_matrix(plot=False)

        # plot heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Matrice de corrélation (spectral)')
        plot_url = _save_fig_to_uploads(fig, prefix='spectral_corr')

        # clustering dendrogram
        Z = analyzer.hierarchical_clustering(plot=False)
        try:
            fig2 = None
            import scipy.cluster.hierarchy as sch
            fig2, ax2 = plt.subplots(figsize=(8,4))
            sch.dendrogram(Z, ax=ax2)
            ax2.set_title('Dendrogramme')
            dendro_url = _save_fig_to_uploads(fig2, prefix='spectral_dendro')
        except Exception:
            dendro_url = None

        return jsonify(success=True, matrix=corr.fillna(0).to_dict(), plot_url=plot_url, dendrogram_url=dendro_url)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/maths/statistical-describe', methods=['POST'])
@app.route('/api/sciences/statistical-describe', methods=['POST'])
def api_maths_statistical_describe():
    try:
        df, _ = _load_dataframe_from_session()
        if StatisticalAnalyzer is None:
            return jsonify(success=False, error="Fonctionnalité indisponible (StatisticalAnalyzer manquant)"), 501
        sa = StatisticalAnalyzer(df)
        desc = df.describe(include='all').to_dict()
        corr = df.select_dtypes(include='number').corr()

        # save corr heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        plot_url = _save_fig_to_uploads(fig, prefix='stat_desc_corr')

        # quick normality test for first numeric column
        numeric = df.select_dtypes(include='number')
        normality = None
        if numeric.shape[1] > 0:
            col0 = numeric.columns[0]
            normality = sa.normality_test(col0)

        return jsonify(success=True, describe=desc, corr=corr.fillna(0).to_dict(), plot_url=plot_url, normality=normality)
    except FileNotFoundError as e:
        return jsonify(success=False, error=str(e)), 400
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/api/export/plot-to-pdf', methods=['POST'])
def api_export_plot_to_pdf():
    try:
        payload = request.get_json() or {}
        plot_url = payload.get('plot_url')
        if not plot_url:
            return jsonify(success=False, error='plot_url is required'), 400
        # accept either full URL path (/uploads/xxx) or filename
        if plot_url.startswith('/uploads/'):
            filename = plot_url.split('/uploads/')[-1]
        else:
            filename = os.path.basename(plot_url)
        src_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(src_path):
            return jsonify(success=False, error='Source image not found'), 404

        base, _ = os.path.splitext(filename)
        out_name = f"{base}_{int(datetime.utcnow().timestamp())}.pdf"
        out_path = os.path.join(UPLOAD_FOLDER, out_name)

        # Try Pillow first
        try:
            from PIL import Image
            img = Image.open(src_path)
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            img.save(out_path, "PDF", resolution=100.0)
        except Exception:
            # Fallback to fpdf
            try:
                from fpdf import FPDF
                pdf = FPDF(unit='mm', format='A4')
                pdf.add_page()
                # Fit image width to page (A4 width ~210mm less margins)
                pdf.image(src_path, x=10, y=10, w=190)
                pdf.output(out_path)
            except Exception as e:
                return jsonify(success=False, error=str(e)), 500

        return jsonify(success=True, pdf_url=f"/uploads/{out_name}")
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500






#===========================================================================
# Lancement de l'app
if __name__ == "__main__":
    app.run(debug=True, port=5002)
#===========================================================================