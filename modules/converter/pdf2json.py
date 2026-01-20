"""
Conversion PDF vers JSON enrichi avec Granite
Pipeline: PDF → JSON (Granite) → Injection images + formules
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _lazy_import_granite():
    # Granite est local au package converter
    from .graniteIBM import granite  # type: ignore

    return granite


def _lazy_import_figindoc():
    # Optionnel: module externe/expérimental
    try:
        from modules.figindoc import extract_images_with_context  # type: ignore

        return extract_images_with_context
    except Exception:
        return None


def pdf2json_enriched(
    pdf_path: str,
    output_dir: Optional[str] = None,
    *,
    extract_images: bool = True,
    preserve_formulas: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convertit un PDF en JSON enrichi.

    - Granite produit le JSON de base
    - Si `figindoc` est dispo, on extrait des images et on les référence dans le JSON
    """

    granite = _lazy_import_granite()
    extract_images_with_context = _lazy_import_figindoc()

    pdf_path_p = Path(pdf_path)
    if not pdf_path_p.exists():
        return {"success": False, "error": f"Fichier introuvable: {pdf_path_p}"}

    output_base = Path(output_dir or "result/output")
    pdf_folder = output_base / pdf_path_p.stem
    pdf_folder.mkdir(parents=True, exist_ok=True)

    images_folder = pdf_folder / "images"
    images_folder.mkdir(exist_ok=True)

    if verbose:
        print(f"[pdf2json] Granite → JSON: {pdf_path_p.name}")

    try:
        produced_json = granite(
            input_path=str(pdf_path_p),
            output_format="json",
            preserve_formulas=preserve_formulas,
        )
        produced_json = Path(produced_json)
        with produced_json.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        return {"success": False, "error": f"Granite: {e}"}

    images_data: Dict[str, Any] = {}
    if extract_images:
        if extract_images_with_context is None:
            if verbose:
                print("[pdf2json] figindoc indisponible: extraction d'images ignorée")
        else:
            try:
                img_result = extract_images_with_context(
                    str(pdf_path_p),
                    output_dir=str(images_folder),
                    verbose=False,
                )
                if isinstance(img_result, dict) and img_result.get("success"):
                    mapping_file = img_result.get("mapping_file")
                    if mapping_file:
                        with Path(mapping_file).open("r", encoding="utf-8") as f:
                            images_data = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"[pdf2json] Erreur extraction images: {e}")

    # Enrichissement (soft)
    page_images: Dict[str, Any] = {}
    if images_data and isinstance(images_data, dict) and "images" in images_data:
        for img in images_data.get("images", []):
            try:
                page = img.get("page")
                page_images.setdefault(page, []).append(
                    {
                        "file": img.get("file"),
                        "size": img.get("size"),
                        "path": f"images/{img.get('file')}",
                    }
                )
            except Exception:
                continue

    formulas_count = 0
    if isinstance(json_data, dict) and "main-text" in json_data:
        for item in json_data.get("main-text", []) or []:
            if isinstance(item, dict) and item.get("name") == "formula":
                formulas_count += 1

    if isinstance(json_data, dict):
        json_data["_enriched"] = {
            "images_extracted": len(images_data.get("images", [])) if isinstance(images_data, dict) else 0,
            "images_by_page": page_images,
            "formulas_detected": formulas_count,
            "output_folder": str(pdf_folder),
        }

    enriched_path = pdf_folder / "document_enriched.json"
    with enriched_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    readme = pdf_folder / "README.txt"
    readme.write_text(
        f"Conversion PDF: {pdf_path_p.name}\n"
        f"{'='*70}\n\n"
        f"- JSON: {enriched_path.name}\n"
        f"- Images: {images_folder.name}/\n\n"
        f"Images extraites: {len(images_data.get('images', [])) if isinstance(images_data, dict) else 0}\n"
        f"Formules détectées: {formulas_count}\n",
        encoding="utf-8",
    )

    return {
        "success": True,
        "pdf_path": str(pdf_path_p),
        "output_folder": str(pdf_folder),
        "json_path": str(enriched_path),
        "images_folder": str(images_folder),
        "images_count": len(images_data.get("images", [])) if isinstance(images_data, dict) else 0,
        "formulas_count": formulas_count,
        "size_kb": round(enriched_path.stat().st_size / 1024, 2),
    }


def convert(pdf_path: str | Path, json_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: PDF -> JSON.

    Par défaut: Granite (JSON) et copie vers `json_path`.
    Si `enriched=True`: produit `document_enriched.json` puis copie vers `json_path`.
    """

    pdf_path_p = Path(pdf_path)
    json_path_p = Path(json_path)
    json_path_p.parent.mkdir(parents=True, exist_ok=True)

    enriched = bool(kwargs.pop("enriched", False))
    preserve_formulas = bool(kwargs.get("preserve_formulas", False))

    if enriched:
        res = pdf2json_enriched(
            str(pdf_path_p),
            output_dir=str(json_path_p.parent),
            extract_images=bool(kwargs.get("extract_images", True)),
            preserve_formulas=preserve_formulas,
            verbose=bool(kwargs.get("verbose", False)),
        )
        if not res.get("success"):
            raise RuntimeError(res.get("error") or "Échec conversion enrichie")
        produced = Path(res["json_path"])
    else:
        granite = _lazy_import_granite()
        produced = Path(
            granite(
                input_path=str(pdf_path_p),
                output_format="json",
                preserve_formulas=preserve_formulas,
            )
        )

    if not produced.exists():
        raise RuntimeError("Granite n'a pas produit de JSON")

    if produced.resolve() != json_path_p.resolve():
        try:
            produced.replace(json_path_p)
        except Exception:
            json_path_p.write_bytes(produced.read_bytes())
    return json_path_p


__all__ = ["pdf2json_enriched", "convert"]


# --- Legacy cassé (gardé pour référence; n'est pas exécuté) ---
_LEGACY = r'''


def pdf2json_enriched(
    pdf_path: str,
    output_dir: Optional[str] = None,
    extract_images: bool = True,
    preserve_formulas: bool = False,  # Désactivé par défaut (bug Granite)
from .graniteIBM import granite
) -> Dict:
try:
    from modules.figindoc import extract_images_with_context  # type: ignore
except Exception:
    extract_images_with_context = None
    """
    Convertit PDF en JSON enrichi avec images et formules.
    Crée un dossier dédié: output_dir/nom_pdf/
    
    Pipeline:
    1. Créer dossier dédié au PDF
    2. Granite génère JSON de base
    3. figindoc extrait images avec contexte → dossier/images/
    4. Injection des liens images dans JSON
    
    Args:
        pdf_path: Chemin PDF
        output_dir: Dossier parent (défaut: result/output)
        extract_images: Extraire et lier images
        preserve_formulas: Préserver formules LaTeX (bug transformers)
        verbose: Logs
        
    Returns:
        Dict avec success, json_path, images_count, output_folder, etc.
    """
    
    pdf_path = Path(pdf_path)
    output_base = Path(output_dir or "result/output")
    
    # Créer dossier dédié au PDF
    pdf_folder = output_base / pdf_path.stem
    pdf_folder.mkdir(parents=True, exist_ok=True)
    
    # Sous-dossiers
    images_folder = pdf_folder / "images"
    images_folder.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CONVERSION ENRICHIE: {pdf_path.name}")
        print(f"  Dossier: {pdf_folder}")
        print(f"{'='*70}")
    
    # Étape 1: Granite génère JSON
    if verbose:
        print("1. Génération JSON avec Granite...")
    
    try:
        json_path = granite(
            input_path=str(pdf_path),
            output_format="json",
            preserve_formulas=preserve_formulas
        )
        json_path = Path(json_path)
        
        # Charger JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
    except Exception as e:
        return {"success": False, "error": f"Granite: {e}"}
    
    # Étape 2: Extraire images avec contexte
    images_data = {}
    if extract_images:
        if verbose:
            print("2. Extraction images avec contexte...")
        
    if extract_images and extract_images_with_context is None:
            img_result = extract_images_with_context(
                str(pdf_path),
                output_dir=str(images_folder),
                verbose=False
            )
            
            if img_result["success"]:
                # Charger mapping images
                mapping_path = Path(img_result["mapping_file"])
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    images_data = json.load(f)
                    
                if verbose:
                    print(f"   ✓ {img_result['images_extracted']} images → {images_folder.name}/")
        except Exception as e:
            if verbose:
                print(f"   ⚠ Erreur extraction images: {e}")
    
    # Étape 3: Enrichir JSON avec images ET formules
    if verbose:
        print("3. Enrichissement JSON...")
    
    # Compter formules dans JSON Granite
    formulas_count = 0
    if "main-text" in json_data:
        for item in json_data.get("main-text", []):
            if item.get("name") == "formula":
                formulas_count += 1
    
    # Index images par page
    page_images = {}
    if images_data and "images" in images_data:
        for img in images_data["images"]:
            page = img["page"]
            if page not in page_images:
                page_images[page] = []
            page_images[page].append({
                "file": img["file"],
                "size": img["size"],
                "path": f"images/{img['file']}"  # Chemin relatif
            })
    
    # Injecter enrichissements
    json_data["_enriched"] = {
        "images_extracted": len(images_data.get("images", [])),
        "images_by_page": page_images,
        "formulas_detected": formulas_count,
        "extraction_method": "figindoc_extended_bbox",
        "output_folder": str(pdf_folder)
    }
    
    # Sauvegarder JSON enrichi dans dossier dédié
    enriched_path = pdf_folder / "document_enriched.json"
    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Créer README dans le dossier
    readme = pdf_folder / "README.txt"
    readme.write_text(f"""
Conversion PDF enrichie: {pdf_path.name}
{'='*70}

Structure:
  - document_enriched.json : JSON avec texte, formules, liens images
  - images/                : Images extraites avec contexte (titres, légendes, axes)
  - images/image_mapping.json : Métadonnées images

Statistiques:
  - Images extraites : {len(images_data.get('images', []))}
  - Formules détectées : {formulas_count}
  - Taille JSON : {enriched_path.stat().st_size / 1024:.1f} KB

Prêt pour conversion LaTeX!
""", encoding='utf-8')
    
    if verbose:
        print(f"\n✓ Conversion réussie!")
        print(f"  Dossier : {pdf_folder}")
        print(f"  JSON    : document_enriched.json ({enriched_path.stat().st_size / 1024:.1f} KB)")
        print(f"  Images  : {len(images_data.get('images', []))} dans images/")
        print(f"  Formules: {formulas_count}")
        print(f"{'='*70}\n")
    
    return {
        "success": True,
        "pdf_path": str(pdf_path),
        "output_folder": str(pdf_folder),
        "json_path": str(enriched_path),
        "images_folder": str(images_folder),
        "images_count": len(images_data.get("images", [])),
        "formulas_count": formulas_count,
        "size_kb": round(enriched_path.stat().st_size / 1024, 2)
    }


if __name__ == "__main__":
    # Test
    result = pdf2json_enriched(
        "data/pdfs/Chapter 3 time series.pdf",
def convert(pdf_path: str | Path, json_path: str | Path, **kwargs) -> Path:
    """Wrapper compatibilité: PDF -> JSON.

    Par défaut, utilise Granite (sortie JSON) et écrit le fichier à l'emplacement demandé.
    """

    pdf_path = Path(pdf_path)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    produced = granite(input_path=str(pdf_path), output_format="json", preserve_formulas=kwargs.get("preserve_formulas", False))
    produced = Path(produced)
    if not produced.exists():
        raise RuntimeError("Granite n'a pas produit de JSON")

    if produced.resolve() != json_path.resolve():
        try:
            produced.replace(json_path)
        except Exception:
            json_path.write_bytes(produced.read_bytes())
    return json_path


__all__ = ["pdf2json_enriched", "convert"]
        verbose=True,
        extract_images=True
    )
    print(f"\nRésultat: {result}")

'''
