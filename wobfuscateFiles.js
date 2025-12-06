// wobfuscate-all.js
//
// Applies Wobfuscator-lite to all JS files in ./dataset
// and writes results into ./output/wobfuscator/

const fs = require('fs');
const path = require('path');
const { wobfuscateSource } = require('./wobfuscator');

const INPUT_DIR = './dataset';
const OUTPUT_DIR = './output/wobfuscator';

function getJsFiles(dir) {
    let results = [];
    const items = fs.readdirSync(dir);

    for (const item of items) {
        const full = path.join(dir, item);
        const stat = fs.statSync(full);

        if (stat.isDirectory()) {
            results = results.concat(getJsFiles(full));
        } else if (full.endsWith('.js')) {
            results.push(full);
        }
    }

    return results;
}

function writeFileSafely(outputPath, data) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, data, 'utf-8');
}

async function wobfuscateAll() {
    const files = getJsFiles(INPUT_DIR);
    console.log(`Found ${files.length} JS files for Wobfuscation.\n`);

    for (const file of files) {
        const relative = path.relative(INPUT_DIR, file);
        const code = fs.readFileSync(file, 'utf-8');

        console.log(`→ Wobfuscating ${relative}`);

        const wobfCode = await wobfuscateSource(code);
        const outPath = path.join(OUTPUT_DIR, relative);

        writeFileSafely(outPath, wobfCode);
    }

    console.log('\n✔ Wobfuscation complete.');
}

wobfuscateAll().catch(err => {
    console.error('Error during Wobfuscation:', err);
    process.exit(1);
});
