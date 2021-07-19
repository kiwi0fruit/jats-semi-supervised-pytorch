# %%
import os
import sqlite3
import pandas as pd
from kiwi_bugfix_typechecker.ipython import display

here = './socionics_db/SOLTI_160_ENG__2019_08_07__NXXXX'
conn = sqlite3.connect(here + '/SOLTI-160w.sqlite')


# %%
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# %%
df_end, df_raw, df_out = tuple(pd.read_sql_query(f"SELECT * FROM {name}", conn)
                               for name in ('DB_end', 'DB_raw', 'DB_out'))
# print(type(df_out['date'][0]))
# display(df_end)


# %%
df_raw_ = df_raw.rename(columns={f"i{i}": f"{i}" for i in range(1, 161)})

map_ = {
    'ILE': 1, 'LII': 2, 'SEI': 3, 'ESE': 4,
    'SLE': 5, 'LSI': 6, 'IEI': 7, 'EIE': 8,
    'SEE': 9, 'ESI': 10, 'ILI': 11, 'LIE': 12,
    'IEE': 13, 'EII': 14, 'SLI': 15, 'LSE': 16,
}
df_out_ = df_out.rename(columns={
    'profile_sigma': 'sigma_of_the_profile', 'cor_halves': 'correl_of_the_halves', 'lvl': 'level',
    'sure': 'confidence', 'type': 'self', 'type_name': 'diagnosis', 'educ': 'education',
    **{k: f'{v}' for k, v in map_.items()}
})

f0 = here + '/db_raw.csv'
if os.path.isfile(f0):
    os.remove(f0)
f1 = here + '/db_out.csv'
if os.path.isfile(f1):
    os.remove(f1)

df_out_.to_csv(f1, index=False)

def int_(s):
    try:
        return int(s)
    except ValueError:
        return 0

df_out_ = pd.read_csv(f1, converters=dict(
        sex=lambda s: 5 if (int(s) == 1) else 1,
        self=lambda s: map_.get(s, -1),
        diagnosis=lambda s: map_[s],
        confidence=int_,
    ))
if os.path.isfile(f1):
    os.remove(f1)

df_raw_ = df_raw_.join(df_out_[['sex', 'age', 'education', 'level', 'self', 'confidence', 'diagnosis']])

df_raw_['id'] += 20000
df_out_['id'] += 20000

display(df_out_)
display(df_raw_)


# %%
df_raw_.to_csv(f0, index=False)
df_out_.to_csv(f1, index=False)
# display(df_out.dtypes)
