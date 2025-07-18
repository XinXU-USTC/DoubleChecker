import pdb
import re
import sympy
import threading
from sympy.parsing.latex import parse_latex
from .parser import strip_string

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = str(final_answer).split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    # 3.0 -> 3
    if final_answer.endswith(".0") and final_answer[:-2].isdigit():
        final_answer = final_answer[:-2]
    # 3.00 -> 3
    if final_answer.endswith(".00") and final_answer[:-3].isdigit():
        final_answer = final_answer[:-3]
    if final_answer.endswith("%") and final_answer[:-1].isdigit():
        final_answer = final_answer[:-1]
    # A -> a
    if final_answer.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
        final_answer = final_answer.lower()
    return final_answer

def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    flag = False    
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True
    
    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True
    
    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False

