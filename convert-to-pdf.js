/**
 * convert-to-pdf.js
 * Script untuk mengkonversi HTML ke PDF menggunakan Puppeteer
 */

const puppeteer = require('puppeteer');
const path = require('path');

async function convertToPDF() {
    console.log('🚀 Memulai konversi HTML ke PDF...');

    const htmlPath = path.join(__dirname, 'docs', 'panduan_cbf_ufw.html');
    const pdfPath = path.join(__dirname, 'docs', 'panduan_cbf_ufw.pdf');

    console.log(`📄 Input: ${htmlPath}`);
    console.log(`📑 Output: ${pdfPath}`);

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();

    // Load HTML file
    await page.goto(`file:///${htmlPath}`, {
        waitUntil: 'networkidle0'
    });

    // Generate PDF
    await page.pdf({
        path: pdfPath,
        format: 'A4',
        printBackground: true,
        margin: {
            top: '20mm',
            right: '15mm',
            bottom: '20mm',
            left: '15mm'
        }
    });

    await browser.close();

    console.log('✅ PDF berhasil dibuat!');
    console.log(`📁 Lokasi: ${pdfPath}`);
}

convertToPDF().catch(err => {
    console.error('❌ Error:', err);
    process.exit(1);
});
