# module raytracer_materials.py

import raytracer as _rt

import sqlite3 as _sq3
_conn = _sq3.connect('raytracing.sqlite')
_c = _conn.cursor()

_c.execute('SELECT * FROM material')
_result = _c.fetchall()
_glob = globals()

# informations exportées par le module
materials = {}
for _m in _result:
    _glob[_m[1]] = materials[_m[1]] = {
        'diffuse': _rt.rgb(*_m[5:8]),
        'ambient': _rt.rgb(*_m[2:5]),
        'specular': _rt.rgb(*_m[8:11]),
        'phong': _m[11],
        'mirror': _rt.rgb(*_m[12:15])
    }
