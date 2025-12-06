/**
 * Obfuscation Pipeline (Default + 3 variants)
 * ------------------------------------------
 * 1. default/        -> DE        (Default preset)
 * 2. deadcode/       -> DE + DI   (dead code injection)
 * 3. cff/            -> DE + CFF  (control flow flattening)
 * 4. split_strings/  -> DE + SS   (string splitting)
 */

const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

// ----------------------------
// CONFIG
// ----------------------------
const INPUT_DIR = './dataset';   // put your original JS files here
const OUTPUT_DIR = './output';   // obfuscated files will be written here

// ----------------------------
// BASE OPTIONS = "Default preset, High performance"
// (copied from the README default preset)
// ----------------------------
const baseDefaultOptions = {
    compact: true,
    controlFlowFlattening: false,
    deadCodeInjection: false,
    debugProtection: false,
    debugProtectionInterval: 0,
    disableConsoleOutput: false,
    identifierNamesGenerator: 'hexadecimal',
    log: false,
    numbersToExpressions: false,
    renameGlobals: false,
    selfDefending: false,
    simplify: true,
    splitStrings: false,
    stringArray: true,
    stringArrayCallsTransform: false,
    stringArrayCallsTransformThreshold: 0.5,
    stringArrayEncoding: [],
    stringArrayIndexShift: true,
    stringArrayRotate: true,
    stringArrayShuffle: true,
    stringArrayWrappersCount: 1,
    stringArrayWrappersChainedCalls: true,
    stringArrayWrappersParametersMaxCount: 2,
    stringArrayWrappersType: 'variable',
    stringArrayThreshold: 0.75,
    unicodeEscapeSequence: false
};

// 1. Default obfuscation (DE)
const defaultOptions = {
    ...baseDefaultOptions
};

// 2. DE + Dead Code Injection (DI)
const deadCodeOptions = {
    ...baseDefaultOptions,
    deadCodeInjection: true,
    deadCodeInjectionThreshold: 0.4
};

// 3. DE + Control Flow Flattening (CFF)
const cffOptions = {
    ...baseDefaultOptions,
    controlFlowFlattening: true,
    controlFlowFlatteningThreshold: 0.75
};

// 4. DE + String Splitting (SS)
const splitStringOptions = {
    ...baseDefaultOptions,
    splitStrings: true,
    splitStringsChunkLength: 10
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

        // --- DE (default) ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'default', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, defaultOptions).getObfuscatedCode()
        );

        // --- DE + DI (dead code injection) ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'deadcode', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, deadCodeOptions).getObfuscatedCode()
        );

        // --- DE + CFF (control flow flattening) ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'cff', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, cffOptions).getObfuscatedCode()
        );

        // --- DE + SS (string splitting) ---
        writeFileSafely(
            path.join(OUTPUT_DIR, 'split_strings', relative),
            JavaScriptObfuscator.obfuscate(sourceCode, splitStringOptions).getObfuscatedCode()
        );
    });

    console.log(`\n✔ Finished obfuscation.`);
}

// Run
obfuscateAll();
