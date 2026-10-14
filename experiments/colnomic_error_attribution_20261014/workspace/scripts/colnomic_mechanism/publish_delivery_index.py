"""Expose the assembled report at the audit root with correct relative links."""
from pathlib import Path
import os,re
from urllib.parse import urlsplit,urlunsplit,unquote
from scripts.colnomic_mechanism.common import ROOT,sha,write

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def main():
    source=BASE/'delivery/REPORT.md'
    def rebase(match):
        target=match.group(1);url=urlsplit(target)
        if url.scheme or url.netloc or not url.path:return match.group(0)
        resolved=(source.parent/unquote(url.path)).resolve()
        assert resolved.is_relative_to(ROOT),target
        relative=os.path.relpath(resolved,BASE)
        return ']('+urlunsplit(('', '',relative,url.query,url.fragment))+')'
    report=re.sub(r'\]\(([^)]+)\)',rebase,source.read_text())
    report+='\n[可浏览HTML与全部照片筛选](delivery/index.html) · [一键冻结复现/只验说明](reproduction_support/README.md)\n'
    report+='\n[最终完整入口说明](reproduction_support/README_FINAL.md) · [原九阶段及35配置证据映射](FINAL_PROTOCOL.json) · [缺失及尚未证明范围](MISSING_AND_LIMITATIONS.json)\n'
    (BASE/'REPORT.md').write_text(report)
    (BASE/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ColNomic 全量审计</title><meta http-equiv="refresh" content="0;url=delivery/index.html">
<a href="delivery/index.html">打开配置—场景—照片—Token索引</a>
<a href="REPORT.md">中文总报告</a><a href="reproduction_support/README.md">复现与只验入口</a></html>''')
    write(BASE/'delivery/root_index_validation.json',dict(status='passed',source_report_SHA256=sha(source),
        root_report_SHA256=sha(BASE/'REPORT.md'),link_rule='delivery-relative targets rebased to audit root; final existence checked by verify_delivery'))


if __name__=='__main__':main()
