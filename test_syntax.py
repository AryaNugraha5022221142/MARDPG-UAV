import py_compile
py_compile.compile("mardpg-uav/scripts/evaluate_multiagent.py", doraise=True)
py_compile.compile("mardpg-uav/mardpg_uav/rendering/live.py", doraise=True)
print("Syntax OK")
