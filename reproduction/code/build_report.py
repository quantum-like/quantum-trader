# Inject ../results/comparison_with_paper.json into report_template.html
# and write ../results/report.html (self-contained page).
import json

data = json.load(open("../results/comparison_with_paper.json"))
tpl = open("report_template.html").read()
html = tpl.replace("__DATA__", json.dumps(data))
open("../results/report.html", "w").write(html)
print(f"wrote ../results/report.html ({len(html) / 1024:.0f} KB)")
