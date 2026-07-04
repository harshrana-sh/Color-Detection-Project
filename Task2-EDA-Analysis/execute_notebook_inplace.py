"""
execute_notebook_inplace.py
-----------------------------
A minimal stand-in for `jupyter nbconvert --execute` for environments
without nbclient/nbconvert installed (e.g. offline sandboxes). Executes
each code cell of EDA_Analysis.ipynb in a single shared namespace, in
order, and writes back:
  - stdout produced by the cell (as a 'stream' output)
  - the repr of the last expression, Jupyter-style auto-display (as an
    'execute_result' output)
  - any matplotlib figures created during the cell (as 'display_data'
    image/png outputs, base64-encoded, embedded directly in the .ipynb)

This is only needed to produce an already-executed notebook for review.
Anyone with a normal Jupyter install can also just open
EDA_Analysis.ipynb and press "Run All" - it does not depend on this
script at all.
"""

import ast
import base64
import io
import json
import sys


def run():
    with open("EDA_Analysis.ipynb") as f:
        nb = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = {}
    exec_count = 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec_count += 1
        source = "".join(cell["source"])
        outputs = []

        stdout_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_buf

        try:
            tree = ast.parse(source)
            last_expr_result_code = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_node = tree.body.pop()
                last_expr_result_code = compile(ast.Expression(last_node.value), "<cell>", "eval")
            exec_code = compile(tree, "<cell>", "exec")

            figs_before = set(plt.get_fignums())
            exec(exec_code, namespace)
            result = None
            if last_expr_result_code is not None:
                result = eval(last_expr_result_code, namespace)

            sys.stdout = old_stdout
            stdout_text = stdout_buf.getvalue()
            if stdout_text:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_text.splitlines(keepends=True),
                })

            if result is not None:
                outputs.append({
                    "output_type": "execute_result",
                    "execution_count": exec_count,
                    "data": {"text/plain": repr(result).splitlines(keepends=True)},
                    "metadata": {},
                })

            figs_after = plt.get_fignums()
            for fignum in figs_after:
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": img_b64, "text/plain": ["<Figure>"]},
                    "metadata": {},
                })
            for fignum in figs_after:
                plt.close(fignum)

        except Exception as e:
            sys.stdout = old_stdout
            outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"{type(e).__name__}: {e}"],
            })
            print(f"[cell {exec_count}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            sys.stdout = old_stdout

        cell["outputs"] = outputs
        cell["execution_count"] = exec_count

    with open("EDA_Analysis.ipynb", "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Executed {exec_count} code cells and wrote outputs back to EDA_Analysis.ipynb")


if __name__ == "__main__":
    run()
