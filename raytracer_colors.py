# module raytracer_colors.py

import raytracer as _rt

import sqlite3 as _sq3
_conn = _sq3.connect('raytracing.sqlite')
_c = _conn.cursor()

_c.execute('SELECT * FROM color')
_result = _c.fetchall()
_glob = globals()

# informations exportées par le module
colors = {}
for _r in _result:
    _glob[_r[1]] = colors[_r[1]] = _rt.rgb(_r[2],_r[3],_r[4])
