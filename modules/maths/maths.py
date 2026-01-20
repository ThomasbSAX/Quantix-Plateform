from __future__ import annotations

import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LoadedModule:
    """Informations sur un module chargé depuis `fonctions/`."""

    public_name: str
    file_path: Path
    module: ModuleType


class Maths:
    """Orchestrateur des fonctions dans `fonctions/`.

    But: rendre *toutes* les fonctions de `fonctions/` appelables via une API uniforme
    (backend web, scripts, etc.) avec modularité et interconnexion.

    Points clés:
    - Chargement dynamique de tous les `.py` du dossier (y compris `2d.py`).
    - Tolérance aux erreurs: un module cassé n'empêche pas les autres de fonctionner.
    - Loader robuste: injecte `from __future__ import annotations` à l'exécution
      pour éviter des imports qui cassent à cause des annotations.
    - Exposition stable: `call('module.fonction', ...)`, `registry()`, `index()`.
    """

    _VIRTUAL_PKG = "maths_fonctions"
    _ALIAS_PKG = "fonctions"

    def __init__(
        self,
        *,
        fonctions_dir: Optional[str | Path] = None,
        eager: bool = True,
        include_private: bool = False,
        force_future_annotations: bool = True,
        exclude_modules: Optional[set[str]] = None,
    ) -> None:
        self._base_dir = Path(__file__).resolve().parent
        self._fonctions_dir = Path(fonctions_dir).resolve() if fonctions_dir else (self._base_dir / "fonctions")
        self._include_private = include_private
        self._force_future_annotations = force_future_annotations

        self._exclude_modules = set(exclude_modules or set())
        # Par défaut, on ignore les modules de visualisation (calcul pur).
        self._exclude_modules.update(
            {
                "visualisation_2d",
                "visualisation_3d",
                "visualisation_distribution",
                "figures",
                "curve",
                "2d",
                "3dinteract",
                "spectral",
            }
        )

        self._loaded: Dict[str, LoadedModule] = {}
        self._proxies: Dict[str, SimpleNamespace] = {}
        self._errors: Dict[str, BaseException] = {}
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._registry_built: bool = False

        self._ensure_virtual_packages()

        if eager:
            self.load_all()

    @property
    def fonctions_dir(self) -> Path:
        return self._fonctions_dir

    @property
    def modules(self) -> Dict[str, ModuleType]:
        return {name: lm.module for name, lm in self._loaded.items()}

    @property
    def errors(self) -> Dict[str, BaseException]:
        return dict(self._errors)

    def discover_files(self) -> List[Path]:
        if not self._fonctions_dir.exists():
            raise FileNotFoundError(f"Dossier introuvable: {self._fonctions_dir}")

        files: List[Path] = []
        for p in sorted(self._fonctions_dir.glob("*.py")):
            if p.name == "__init__.py":
                continue
            if p.name.startswith("_") and not self._include_private:
                continue
            if p.stem in self._exclude_modules:
                continue
            files.append(p)
        return files

    def list_modules(self) -> List[str]:
        return [p.stem for p in self.discover_files()]

    def load_all(self) -> None:
        self._ensure_virtual_packages()
        for file_path in self.discover_files():
            public_name = file_path.stem
            if public_name in self._loaded or public_name in self._errors:
                continue
            self._load_one(public_name, file_path)

        self._registry_built = False

    def reload(self, module_name: Optional[str] = None) -> None:
        if module_name is None:
            self._loaded.clear()
            self._proxies.clear()
            self._errors.clear()
            self._registry.clear()
            self._registry_built = False
            self.load_all()
            return

        self._loaded.pop(module_name, None)
        self._proxies.pop(module_name, None)
        self._errors.pop(module_name, None)
        self._registry_built = False

        file_path = self._fonctions_dir / f"{module_name}.py"
        if not file_path.exists():
            raise KeyError(f"Module inconnu: {module_name}")
        self._load_one(module_name, file_path)

    def index(self) -> Dict[str, Any]:
        """Index "prêt backend" (liste modules/fonctions/erreurs)."""
        self.load_all()
        functions = self.list_functions()
        return {
            "fonctions_dir": str(self._fonctions_dir),
            "modules_loaded": sorted(self.modules.keys()),
            "modules_failed": {
                name: f"{type(exc).__name__}: {exc}" for name, exc in sorted(self._errors.items())
            },
            "functions": functions,
        }

    def search(self, query: str, *, limit: int = 50, in_doc: bool = True) -> List[Dict[str, Any]]:
        """Recherche rapide de fonctions par nom/doc.

        Retourne une liste triée de résultats avec score + signature.
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        tokens = [t for t in re.split(r"[\s._\-/]+", q) if t]
        reg = self.registry()
        hits: List[Tuple[int, str, Callable[..., Any]]] = []

        for dotted, fn in reg.items():
            hay = dotted.lower()
            score = 0

            if q in hay:
                score += 100
            for tok in tokens:
                if tok in hay:
                    score += 20

            if in_doc:
                doc = (inspect.getdoc(fn) or "").lower()
                if q in doc:
                    score += 30
                for tok in tokens:
                    if tok in doc:
                        score += 5

            if score > 0:
                hits.append((score, dotted, fn))

        hits.sort(key=lambda t: (-t[0], t[1]))
        out: List[Dict[str, Any]] = []
        for score, dotted, fn in hits[: max(1, int(limit))]:
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            out.append(
                {
                    "name": dotted,
                    "score": score,
                    "signature": sig,
                    "doc": (inspect.getdoc(fn) or "")[:400],
                }
            )
        return out

    def list_functions(self, module: Optional[str] = None) -> List[str]:
        if module is not None:
            mod = self._get_module(module)
            return [f"{module}.{name}" for name in self._public_callables(mod).keys()]

        out: List[str] = []
        for module_name in sorted(self.modules.keys()):
            try:
                out.extend(self.list_functions(module_name))
            except Exception:
                continue
        return out

    def registry(self) -> Dict[str, Callable[..., Any]]:
        """Registre exhaustif: `module.fonction` → callable."""
        if not self._registry_built:
            self.load_all()
            reg: Dict[str, Callable[..., Any]] = {}
            for module_name, mod in self.modules.items():
                for fn_name, fn in self._public_callables(mod).items():
                    reg[f"{module_name}.{fn_name}"] = fn
            self._registry = reg
            self._registry_built = True
        return dict(self._registry)

    def resolve(self, dotted_name: str) -> Callable[..., Any]:
        module_name, attr = self._split_dotted(dotted_name)
        mod = self._get_module(module_name)
        fn = getattr(mod, attr, None)
        if fn is None or not callable(fn):
            raise AttributeError(f"Callable introuvable: {dotted_name}")
        return fn

    def describe(self, dotted_name: str) -> Dict[str, Any]:
        fn = self.resolve(dotted_name)
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"

        return {
            "name": dotted_name,
            "signature": sig,
            "doc": inspect.getdoc(fn) or "",
            "callable_type": "class" if inspect.isclass(fn) else "function",
            "module": getattr(fn, "__module__", ""),
        }

    def call(self, dotted_name: str, /, *args: Any, **kwargs: Any) -> Any:
        fn = self.resolve(dotted_name)
        return fn(*args, **kwargs)

    def call_spec(self, spec: Dict[str, Any]) -> Any:
        """Appel orienté web/JSON.

        spec:
        - name: "module.fonction"
        - args: list
        - kwargs: dict
        """
        if "name" not in spec:
            raise ValueError("spec doit contenir la clé 'name'")
        name = spec["name"]
        args = spec.get("args", [])
        kwargs = spec.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("spec.args doit être une liste et spec.kwargs un dict")
        return self.call(name, *args, **kwargs)

    def call_spec_json(self, spec_json: str) -> Any:
        return self.call_spec(json.loads(spec_json))

    def call_json(self, dotted_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Appel + conversion en objets sérialisables JSON (utile Flask)."""
        return self._to_jsonable(self.call(dotted_name, *args, **kwargs))

    def call_spec_jsonable(self, spec: Dict[str, Any]) -> Any:
        """Comme call_spec(), mais renvoie un résultat JSON-safe."""
        return self._to_jsonable(self.call_spec(spec))

    def call_spec_json_jsonable(self, spec_json: str) -> Any:
        return self.call_spec_jsonable(json.loads(spec_json))

    def get(self, module_name: str) -> ModuleType:
        return self._get_module(module_name)

    def __getattr__(self, name: str) -> Any:
        # Permet: maths.calcul_arithmetique.addition(...)
        if name in self._proxies:
            return self._proxies[name]

        if name not in self._loaded and name not in self._errors:
            file_path = self._fonctions_dir / f"{name}.py"
            if file_path.exists():
                self._load_one(name, file_path)
                self._registry_built = False

        if name in self._loaded:
            proxy = self._build_proxy(name, self._loaded[name].module)
            self._proxies[name] = proxy
            return proxy

        raise AttributeError(name)

    # -------------------------
    # Internes
    # -------------------------

    def _ensure_virtual_packages(self) -> None:
        # Package virtuel principal.
        pkg = sys.modules.get(self._VIRTUAL_PKG)
        if pkg is None:
            pkg = ModuleType(self._VIRTUAL_PKG)
            pkg.__path__ = [str(self._fonctions_dir)]  # type: ignore[attr-defined]
            sys.modules[self._VIRTUAL_PKG] = pkg

        # Alias optionnel: `import fonctions.xxx`.
        alias = sys.modules.get(self._ALIAS_PKG)
        if alias is None:
            alias = ModuleType(self._ALIAS_PKG)
            alias.__path__ = [str(self._fonctions_dir)]  # type: ignore[attr-defined]
            sys.modules[self._ALIAS_PKG] = alias

    def _split_dotted(self, dotted_name: str) -> Tuple[str, str]:
        parts = dotted_name.split(".")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Nom invalide, attendu: 'module.fonction'")
        return parts[0], parts[1]

    def _safe_module_stem(self, stem: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in stem)
        if not safe or safe[0].isdigit():
            safe = f"_{safe}"
        return safe

    def _internal_module_name(self, public_name: str) -> str:
        safe = self._safe_module_stem(public_name)
        return f"{self._VIRTUAL_PKG}.{safe}"

    def _load_one(self, public_name: str, file_path: Path) -> None:
        self._ensure_virtual_packages()
        module_name = self._internal_module_name(public_name)
        safe = module_name.split(".", 1)[1]

        try:
            module = ModuleType(module_name)
            module.__file__ = str(file_path)
            module.__package__ = self._VIRTUAL_PKG
            module.__dict__["__file__"] = str(file_path)
            module.__dict__.setdefault("MATHS", self)

            source = file_path.read_text(encoding="utf-8")
            if self._force_future_annotations and "from __future__ import annotations" not in source:
                source = "from __future__ import annotations\n" + source

            code = compile(source, str(file_path), "exec", dont_inherit=True)
            sys.modules[module_name] = module
            # Alias d'import stable
            sys.modules[f"{self._ALIAS_PKG}.{safe}"] = module
            exec(code, module.__dict__)

            self._loaded[public_name] = LoadedModule(public_name=public_name, file_path=file_path, module=module)
        except BaseException as exc:
            self._errors[public_name] = exc
            sys.modules.pop(module_name, None)
            sys.modules.pop(f"{self._ALIAS_PKG}.{safe}", None)

    def _get_module(self, module_name: str) -> ModuleType:
        if module_name not in self._loaded:
            if module_name not in self._errors:
                file_path = self._fonctions_dir / f"{module_name}.py"
                if file_path.exists():
                    self._load_one(module_name, file_path)
                    self._registry_built = False

        if module_name in self._loaded:
            return self._loaded[module_name].module

        if module_name in self._errors:
            raise ImportError(f"Import échoué pour {module_name}: {self._errors[module_name]}")

        raise KeyError(f"Module inconnu: {module_name}")

    def _public_callables(self, module: ModuleType) -> Dict[str, Callable[..., Any]]:
        out: Dict[str, Callable[..., Any]] = {}
        for name, value in vars(module).items():
            if name.startswith("_") and not self._include_private:
                continue
            if not callable(value):
                continue

            module_of_obj = getattr(value, "__module__", None)
            if module_of_obj != module.__name__:
                continue

            if inspect.isfunction(value) or inspect.isclass(value):
                out[name] = value

        return out

    def _build_proxy(self, module_name: str, module: ModuleType) -> SimpleNamespace:
        public = self._public_callables(module)
        proxy = SimpleNamespace(**public)
        setattr(proxy, "__module__", module)
        setattr(proxy, "__name__", module_name)
        return proxy

    def _to_jsonable(self, obj: Any) -> Any:
        """Convertit récursivement vers des types sérialisables JSON.

        - numpy.ndarray -> list
        - numpy scalars -> python scalars
        - pandas -> dict/list (si dispo)
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # numpy
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
        except Exception:
            pass

        # pandas
        try:
            import pandas as pd  # type: ignore

            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_list()
        except Exception:
            pass

        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]

        # fallback (ex: dataclasses / objects)
        if hasattr(obj, "__dict__"):
            return self._to_jsonable(vars(obj))

        return str(obj)
