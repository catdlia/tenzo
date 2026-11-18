#!/usr/bin/env python3
"""Count vector register spills/reloads inside loops of x86 AT&T assembly (llc .s output or llvm-objdump -d output).

A spill = vector store to a stack slot (%rsp/%rbp-relative), a reload = vector load from one. Loops are regions closed by a backward
branch. We report every innermost loop (no nested backward branch) and optionally write its body for llvm-mca.
No numbers are hard-coded: everything is counted from the input files.
"""
import argparse, re, sys
from pathlib import Path

VMOV = re.compile(r'^\s*(?:[0-9a-f]+:\s*)?(v?mov(?:ups|aps|dqu|dqa|dqu8|dqa64|dqu64|ss|sd|d|q)|vbroadcast\w+|vpbroadcast\w+)\s+(.*)$')
STACK = re.compile(r'-?\d*\(%(?:rsp|rbp)(?:,[^)]*)?\)')
FMA = re.compile(r'\bvfn?m(?:add|sub)\d+p[sd]\b|\bvpmaddwd\b|\bvpmaddubsw\b|\bvpdpbusd\b|\bvpdpwssd\b')
JMP = re.compile(r'^\s*(?:[0-9a-f]+:\s*)?(j\w+)\s+(\S+)')
LABEL = re.compile(r'^(\.?[A-Za-z_][\w.$]*):')


def parse(path):
    """-> list of (index, kind, text, label_or_addr); kind in {'ins','label'}"""
    out, is_dis = [], False
    for ln in Path(path).read_text().splitlines():
        s = ln.rstrip()
        if not s or s.lstrip().startswith(('#', '.')) and not LABEL.match(s.lstrip()):
            if not LABEL.match(s):
                continue
        m = LABEL.match(s)
        if m and not re.match(r'^\s*[0-9a-f]+:\s', s):
            out.append(('label', s, m.group(1)))
            continue
        m = re.match(r'^\s*([0-9a-f]+):\s+(.*)$', s)               # objdump line:  "  4a2c: vfmadd231ps ..."
        if m:
            is_dis = True
            out.append(('ins', m.group(2).strip(), m.group(1)))
        elif s.startswith(('\t', ' ')):
            out.append(('ins', s.strip(), None))
    return out, is_dis


def loops(items, is_dis):
    pos = {}
    for i, (k, t, tag) in enumerate(items):
        if k == 'label': pos[tag] = i
        elif is_dis and tag: pos[tag.lstrip('0') or '0'] = i
    regs = []
    for i, (k, t, tag) in enumerate(items):
        if k != 'ins': continue
        m = JMP.match(t)
        if not m: continue
        tgt = m.group(2).lstrip('*').split()[0].rstrip(',')
        key = tgt if not is_dis else tgt.lstrip('0x').lstrip('0') or '0'
        j = pos.get(tgt) if not is_dis else pos.get(key)
        if j is not None and j <= i and m.group(1) != 'call':
            regs.append((j, i))
    return regs


def stats(items, a, b):
    ins = [t for k, t, _ in items[a:b + 1] if k == 'ins']
    n_st = n_ld = n_fma = n_vec = 0
    for t in ins:
        m = VMOV.match(t)
        if m and STACK.search(m.group(2)):
            ops = m.group(2).split(',')
            # AT&T: last operand is the destination. memory destination => store (spill), memory source => load (reload)
            if STACK.search(ops[-1]): n_st += 1
            else: n_ld += 1
        if FMA.search(t): n_fma += 1
        if re.search(r'%[xyz]mm\d+', t): n_vec += 1
    return len(ins), n_vec, n_fma, n_st, n_ld, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--md', help='write markdown here'); ap.add_argument('--mca-dir', help='write innermost hot loop bodies here')
    a = ap.parse_args()
    md = ['| file | whole-function vec spills (st/ld) | innermost loop | instrs | vec instrs | FMA/madd | spill st | reload ld | spill+reload per FMA |', '|---|---|---|---|---|---|---|---|---|']
    for f in a.files:
        items, is_dis = parse(f)
        tot = stats(items, 0, len(items) - 1)
        regs = loops(items, is_dis)
        inner = [(x, y) for (x, y) in regs if not any((x < u and v < y) or (x <= u and v < y) or (x < u and v <= y) for (u, v) in regs if (u, v) != (x, y))]
        inner = sorted(set(inner))
        name = Path(f).name
        if not inner:
            md.append(f'| {name} | {tot[3]}/{tot[4]} | (no loop found) | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} | - |')
            continue
        best = None
        for idx, (x, y) in enumerate(inner):
            n, v, fm, st, ld, ins = stats(items, x, y)
            ratio = (st + ld) / fm if fm else float('nan')
            md.append(f'| {name} | {tot[3]}/{tot[4]} | loop {idx} [{x}:{y}] | {n} | {v} | {fm} | {st} | {ld} | {ratio:.2f} |' if fm else
                      f'| {name} | {tot[3]}/{tot[4]} | loop {idx} [{x}:{y}] | {n} | {v} | 0 | {st} | {ld} | - |')
            if a.mca_dir and (best is None or fm > best[0]) and fm:
                best = (fm, idx, ins)
        if a.mca_dir and best:
            Path(a.mca_dir).mkdir(parents=True, exist_ok=True)
            body = [re.sub(r'\s+#.*$', '', t) for t in best[2] if not JMP.match(t)]
            Path(a.mca_dir, name + f'.loop{best[1]}.s').write_text('\n'.join(body) + '\n')
    text = '\n'.join(md) + '\n'
    if a.md: Path(a.md).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
