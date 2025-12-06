/**
 * Obfuscation Pipeline (4 Separate Modes)
 * ---------------------------------------
 * 1. default/
 * 2. deadcode/
 * 3. cff/           control-flow-flattening ONLY
 * 4. split_strings/ string splitting ONLY
 */

const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

// ----------------------------
// CONFIG
// ----------------------------
const INPUT_DIR = './dataset';   // put your JS files here
const OUTPUT_DIR = './output';   // obfuscated files go here

// ----------------------------
// OBFUSCATION OPTIONS
// ----------------------------

// 1. DEFAULT
const defaultOptions = {
    compact: true,
    controlFlowFlattening: false,
    deadCodeInjection: false,
    splitStrings: false,
    stringArray: true,
};

// 2. DEAD CODE INJECTION ONLY
const deadCodeOptions = {
    compact: true,
    deadCodeInjection: true,
    deadCodeInjectionThreshold: 1,
    controlFlowFlattening: false,
    splitStrings: false,
    stringArray: true,
};

// 3. CONTROL FLOW FLATTENING ONLY
const cffOptions = {
    compact: true,
    controlFlowFlattening: true,
    controlFlowFlatteningThreshold: 1,
    deadCodeInjection: false,
    splitStrings: false,
    stringArray: true,
};

// 4. STRING SPLITTING ONLY
const splitStringOptions = {
    compact: true,
    splitStrings: true,
    splitStringsChunkLength: 5,
    controlFlowFlattening: false,
    deadCodeInjection: false,
    stringArray: true,
};

// ----------------------------
// SCAN JS FILES
// ----------------------------
function getJsFiles(dir) {
    let results = [];
    const list = fs.readdirSync(dir);

    list.forEach((file) => {
        const filepath = path.join(dir, file);
        const stat = fs.statSync(filepath);

        if (stat && stat.isDirectory()) {
            results = results.concat(getJsFiles(filepath));
        } else if (filepath.endsWith('.js')) {
            results.push(filepath);
        }
    });

    return results;
}

// ----------------------------
// Write file with safe directory creation
// ----------------------------
function writeFileSafely(outputPath, data) {
    const dir = path.dirname(outputPath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(outputPath, data, 'utf-8');
}

// ----------------------------
// MAIN PIPELINE
// ----------------------------
function obfuscateAll() {
    const jsFiles = getJsFiles(INPUT_DIR);

    console.log(`Found ${jsFiles.length} JS files to obfuscate.\n`);

    jsFiles.forEach((fullPath) => {
        const relative = path.relative(INPUT_DIR, fullPath);
        const sourceCode = fs.readFileSync(fullPath, 'utf-8');

        console.log(`→ Obfuscating ${relative}`);

        // --- default ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'default', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, defaultOptions).getObfuscatedCode()
        );

        // --- deadcode only ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'deadcode', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, deadCodeOptions).getObfuscatedCode()
        );

        // --- control flow flattening only ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'cff', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, cffOptions).getObfuscatedCode()
        );

        // --- string splitting only ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'split_strings', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, splitStringOptions).getObfuscatedCode()
        );
    });

    console.log(`\n✔ Finished obfuscation.`);
}

// Run
obfuscateAll();
