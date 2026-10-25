// Render report/report.html to report/report.pdf with the pre-installed Chromium (Playwright).
// Usage: node report/make_pdf.js   (fonts and MathJax load from their CDNs; figures are local files)
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const dir = __dirname;
  const html = 'file://' + path.join(dir, 'report.html');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } });
  await page.goto(html, { waitUntil: 'networkidle' });
  // wait for MathJax typesetting if it loaded
  await page.evaluate(() => (window.MathJax && MathJax.startup && MathJax.startup.promise) ? MathJax.startup.promise : null);
  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: path.join(dir, 'report.pdf'), format: 'A4', printBackground: true,
    margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:8px;color:#777;width:100%;text-align:center;font-family:sans-serif">Hard-Sphere Electrolyte Movies — page <span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  // also a quick screenshot of the top of the page for a visual check
  await page.emulateMedia({ media: 'screen' });
  await page.screenshot({ path: path.join(dir, 'data', 'preview_top.png'), fullPage: false });
  await page.evaluate(() => { const el = document.getElementById('r-order'); if (el) el.scrollIntoView(); });
  await page.screenshot({ path: path.join(dir, 'data', 'preview_order.png'), fullPage: false });
  await page.setViewportSize({ width: 400, height: 900 });
  await page.evaluate(() => { const el = document.getElementById('r-pairing'); if (el) el.scrollIntoView(); });
  await page.screenshot({ path: path.join(dir, 'data', 'preview_phone.png'), fullPage: false });
  await browser.close();
  console.log('wrote', path.join(dir, 'report.pdf'));
})().catch(e => { console.error(e); process.exit(1); });
