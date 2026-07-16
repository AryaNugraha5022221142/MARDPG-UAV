import re

with open('mardpg-uav/scripts/evaluate_multiagent.py', 'r') as f:
    content = f.read()

# I also added traceback printing to the exception earlier so the user might see other errors, let's remove it to keep it clean.
content = re.sub(
    r'except Exception as ex:\n                            import traceback\n                            traceback.print_exc\(\)',
    'except Exception as ex:\n                            pass',
    content
)

with open('mardpg-uav/scripts/evaluate_multiagent.py', 'w') as f:
    f.write(content)
