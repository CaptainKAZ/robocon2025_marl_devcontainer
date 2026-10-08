#!/usr/bin/env node
/* 验收（无浏览器）：从 live_view.html 抽出 CODE_META / CODE_ORDER / codeOn 与 drawCurve 的那行循环，
 * 在 Node 里用桩 document 跑一遍，确认 10 个码都能出曲线、标签与取值正确。
 * 用法: node .opencode/check_curve_logic.js
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'BenchMARL', 'liveview', 'live_view.html'), 'utf8');

function grab(re, name) {
  const m = html.match(re);
  if (!m) { console.error(`FAIL 未能在 HTML 中找到 ${name}`); process.exit(1); }
  return m[0];
}

const codeMetaSrc = grab(/const CODE_META = \{[\s\S]*?\n\};/, 'CODE_META');
const codeOrderSrc = grab(/const CODE_ORDER = [\s\S]*?;/, 'CODE_ORDER');
const codeOnSrc = grab(/function codeOn\(n\) \{[^\n]*\}/, 'codeOn');
const loopSrc = grab(/for \(const n of CODE_ORDER\) if \(codeOn\(n\)\) \{[^\n]*\}/, 'drawCurve 循环行');

const checked = new Set([1, 2, 3, 4, 5, 11, 12, 13, 14, 15]);
global.document = { getElementById: (id) => ({ checked: checked.has(Number(id.replace('code', ''))) }) };

const point = { x: 7, c1: 11, c2: 22, c3: 3.3, c4: 4.4, c5: 0.5, c11: 6.6, c12: 7.7, c13: 8.8, c14: 0.9, c15: 1.1 };
const ds = [];
const mk = (label, color, get) => ({ label, y: get(point) });

eval(codeMetaSrc + codeOrderSrc + codeOnSrc + loopSrc);

const labels = ds.map((d) => d.label);
const fails = [];
if (ds.length !== 10) fails.push(`曲线条数 ${ds.length} != 10`);
for (const n of [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]) {
  const item = ds.find((d) => d.label.startsWith('码' + n + ' '));
  if (!item) fails.push(`缺 码${n} 曲线`);
  else if (Math.abs(item.y - point['c' + n]) > 1e-9) fails.push(`码${n} 取值 ${item.y} != ${point['c' + n]}`);
}

console.log('曲线标签:', labels.join(' | '));
if (fails.length) { console.error('FAIL\n - ' + fails.join('\n - ')); process.exit(1); }
console.log('PASS 10 个结束原因全部成曲线，标签与取值正确');
