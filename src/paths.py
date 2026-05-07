from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / 'data'

RAW_DATA = DATA / 'raw'
INTERIM_DATA = DATA / 'interim'
CLEAN_DATA = DATA / 'clean'

OUTPUT = ROOT / 'output'
FIG_OUTPUT = OUTPUT / 'fig'
