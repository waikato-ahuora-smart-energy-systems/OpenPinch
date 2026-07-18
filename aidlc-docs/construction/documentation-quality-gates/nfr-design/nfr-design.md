# NFR Design

The CSV manifest is checked against live owner introspection and notebook ASTs
in pytest, and rendered directly by Sphinx. Notebook profiles remain metadata,
not implicit skip claims: base execution is mandatory, while optional profiles
are invoked by explicit quality-gate commands. Distribution verification builds
fresh archives and compares their notebook inventory with the manifest.
