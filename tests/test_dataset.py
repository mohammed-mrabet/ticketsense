import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "generate_dataset", Path(__file__).resolve().parents[1] / "data" / "generate_dataset.py"
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_generation_reproductible():
    assert gen.generate(200, seed=1) == gen.generate(200, seed=1)


def test_colonnes_et_valeurs():
    rows = gen.generate(300)
    assert len(rows) == 300
    assert {r["category"] for r in rows} <= set(gen.CATEGORIES)
    assert {r["priority"] for r in rows} <= set(gen.PRIORITIES)
    assert all(r["body"] for r in rows)
    # Un ticket résolu l'est toujours après sa création.
    assert all(not r["resolved_at"] or r["resolved_at"] >= r["created_at"] for r in rows)
