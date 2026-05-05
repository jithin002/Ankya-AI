path = 'test_extraction.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('\u2500', '-'),
    ('\u2502', '|'),
    ('\u2190', '<-'),
    ('\u2192', '->'),
    ('\u2248', '~'),
    ('\u251c', '|'),
    ('\u2514', '+'),
    ('\u2550', '='),
    ('\u2018', "'"),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done - all unicode box/arrow chars replaced with ASCII equivalents')
