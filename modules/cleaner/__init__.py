"""Module central du cleaner.

Expose `Cleaner` (toujours disponible) et, si possible, des outils optionnels
(Predictor, CSVEditor, etc.) sans casser l'import du package.
"""

from .Cleaner import Cleaner

__all__ = ["Cleaner"]

try:
	from .Predictor import Predictor, generate_prediction_report

	__all__ += ["Predictor", "generate_prediction_report"]
except Exception:
	Predictor = None
	generate_prediction_report = None

try:
	from .CSVEditor import CSVEditor

	__all__ += ["CSVEditor"]
except Exception:
	CSVEditor = None
